import torch
import torch.nn.functional as F
from torch import nn
from models.WMSM_Net_Backbone import MSC_WMSM_NetBackbone, BSPSWT
from einops import rearrange
from torchsummary import summary
import math
from WMSM_Net_spectral import WMSM_Net_Spectral
from sklearn.cluster import KMeans  

# models/WMSM_Net_Backbone.py

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  
            nn.ReLU(inplace=True),  
            nn.Linear(channel // reduction, channel, bias=False),  
            nn.Sigmoid()  
        )

    def forward(self, x):
        b, c, _, _ = x.size()  
        y = self.avg_pool(x).view(b, c)  
        y = self.fc(y).view(b, c, 1, 1)  
        return x * y.expand_as(x)  



class Frequency_Spectrum_Dynamic_Aggregation(nn.Module):
    
    def __init__(self, nc):
        super(Frequency_Spectrum_Dynamic_Aggregation, self).__init__()  
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),  
            nn.LeakyReLU(0.1, inplace=True),  
            SELayer(channel=nc),  
            nn.Conv2d(nc, nc, 1, 1, 0)  
        )
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),  
            nn.LeakyReLU(0.1, inplace=True),  
            SELayer(channel=nc),  
            nn.Conv2d(nc, nc, 1, 1, 0)  
        )

    
    def forward(self, x):
        _, _, H, W = x.shape  
        # input = torch.rand(4, 32, 64, 64)

        
        
        x_freq = torch.fft.rfft2(x, norm='backward')  

        ori_mag = torch.abs(x_freq)  
        mag = self.processmag(ori_mag)  
        mag = ori_mag + mag  

        ori_pha = torch.angle(x_freq)  
        pha = self.processpha(ori_pha)  
        pha = ori_pha + pha  

        real = mag * torch.cos(pha)  
        imag = mag * torch.sin(pha)  
        x_out = torch.complex(real, imag)  

        x_freq_spatial = torch.fft.irfft2(x_out, s=(H, W), norm='backward')  
        return x_freq_spatial  


class WMSM_Net(nn.Module):
    def __init__(self, num_band, d_model=256, num_endm=3, num_queries_times=30, 
                 scale=3.5, ds=4, dropout=5e-2, vca_endmember=None):  
        super(WMSM_Net, self).__init__()
        self.num_band = num_band
        self.num_endm = num_endm
        self.down_ratio = ds
        self.num_queries_times = num_queries_times
        
        
        self.bspswt = BSPSWT(wavelet='bior3.3', level=1)
        #self.bspswt = BSPDWT(wavelet='bior3.3', level=1)
        self.gcs_feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, H, W] → [B, C, 1, 1]
            nn.Conv2d(in_channels=num_band, out_channels=num_band, kernel_size=1, bias=False),  
            nn.GELU()
        )

        
        self.backbone = MSC_WMSM_NetBackbone(in_channels=num_band, hidden_dim=d_model, num_classes=num_endm,
                                        scale=scale, ds=ds, dropout=dropout)
        
        self.num_queries = num_queries_times * num_endm
        
        
        if vca_endmember is not None:
            
            
            vca_endmember = torch.from_numpy(vca_endmember).float()  
            
            vca_repeated = vca_endmember.T.unsqueeze(1).repeat(1, num_queries_times, 1)
            
            vca_flat = vca_repeated.reshape(-1, num_band)
            self.query_embed = nn.Embedding.from_pretrained(vca_flat, freeze=False)  
        else:
            
            self.query_embed = nn.Embedding(self.num_queries, num_band)
        
        self.weights = nn.Parameter(torch.ones((num_endm, num_queries_times)))
        self._reset_parameters()
        
       
        self.kernel_size = 3
        self.pad = self.kernel_size // 2  
        
        self.use_abu_sim = None
        
        
        
        #self.freq_agg = Frequency_Spectrum_Dynamic_Aggregation(nc=num_band)
        self.freq_agg_LF = Frequency_Spectrum_Dynamic_Aggregation(nc=num_band)
        self.freq_agg_HF = Frequency_Spectrum_Dynamic_Aggregation(nc=num_band)
        self.freq_agg_VF = Frequency_Spectrum_Dynamic_Aggregation(nc=num_band)
        self.freq_agg_DF = Frequency_Spectrum_Dynamic_Aggregation(nc=num_band)        
        
        
        self.sfe = nn.ModuleDict({
            "conv3d_features": nn.Sequential(
                nn.Conv3d(1, 8, (5, 1, 1), 1),   
                nn.BatchNorm3d(8),
                nn.LeakyReLU(inplace=True),
            ),
            "conv2d_features": nn.Sequential(
                nn.Conv2d(8 * (self.num_band - 4), self.num_band, 1, 1),  
                nn.BatchNorm2d(self.num_band),
                nn.LeakyReLU(inplace=True),
            )
        })
        
         
        
        self.gcs_scale = nn.Parameter(torch.tensor(0.1))
    def extract_global_spectral_feat(self, lf_feature):

          B, C, H, W = lf_feature.shape
          
          x = lf_feature.unsqueeze(1)  # [B, 1, C, H, W]
          
          
          x_3d = self.sfe["conv3d_features"](x)  # [B, 8, C-4, H, W]
          
          
          x_2d = x_3d.view(B, -1, H, W)  
          
          
          x_2d = self.sfe["conv2d_features"](x_2d)  # [B, num_band, H, W]
          
          
          global_feat = F.adaptive_avg_pool2d(x_2d, 1).squeeze(-1).squeeze(-1)  # [B, num_band]
          
          
          
          global_feat = F.normalize(global_feat, p=2, dim=1)
          return global_feat

    ""
    def compute_gcs_matrix(self, global_spe_feat, pred_endm):
        ""
        
        pred_endm_norm = F.normalize(pred_endm, p=2, dim=1) # [num_endm, num_band]
        global_spe_feat_norm = global_spe_feat.squeeze(0)   # [num_band]
        
        
        
        
        phi = torch.einsum('ec, c -> e', pred_endm_norm, global_spe_feat_norm) 
        
        
        
        gcs_matrix = phi.unsqueeze(-1).repeat(1, self.num_band)
        
        
        
        
        
        gcs_matrix = torch.tanh(gcs_matrix)
        
        return gcs_matrix                  
    
    def median_filter(self, x):
            ""
            B, C, H, W = x.shape
            
            
            x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
            
            
            x_unfold = x_pad.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
            
            
            x_unfold = x_unfold.contiguous().view(B, C, H, W, -1)  # [B, C, H, W, 9]
            x_median = x_unfold.median(dim=-1)[0]  # [B, C, H, W]
            
            return x_median
            
    def euclidean_sim(self, data):
        norm_data = torch.sum(data ** 2, dim=1).reshape(-1, 1)
        dist_matrix = norm_data - 2 * torch.matmul(data, data.t()) + norm_data.t()
        similarity_matrix = torch.exp(-dist_matrix)
        return similarity_matrix

    def mat2gray(self, Y):
        min_val = torch.min(Y)
        max_val = torch.max(Y)
        return (Y - min_val) / (max_val - min_val + 1e-8)

    def abu_similarity(self, data, c=0.5):
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        y = data.squeeze(0)
        b, h, w = y.size()
        spa = y.view(b, -1)
        spaT = spa.permute(1, 0)
        S1 = self.euclidean_sim(spa)
        S2 = self.euclidean_sim(spaT)
        y_flat = y.view(b, h * w)
        y1 = torch.matmul(S1, y_flat)
        y2 = torch.matmul(y_flat, S2)
        y1 = self.mat2gray(y1)
        y2 = self.mat2gray(y2)
        y = c * y1 + (1 - c) * y2
        y = self.mat2gray(y)
        y = y.view(b, h, w).unsqueeze(0)
        return y

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight) 
    
 
    def forward(self, x):
        
        LF, HF, VF, DF = self.bspswt(x)
        
        #assert LF.shape[1] == self.num_band, \
        
        
        
        
        #freq_agg = Frequency_Spectrum_Dynamic_Aggregation(nc=self.num_band).to(x.device)
        
        
        
        
        
        
        LF = self.freq_agg_LF(LF)
        HF = self.freq_agg_HF(HF)
        VF = self.freq_agg_VF(VF)
        DF = self.freq_agg_DF(DF)
        
                        
        
        endm_get = self.get_endmember(lf_feature=LF)
        
        
        _, _, w, h = x.shape
        pad_1 = math.ceil(w / self.down_ratio) * self.down_ratio - w
        pad_2 = math.ceil(h / self.down_ratio) * self.down_ratio - h
        LF_patch = F.pad(LF, (0, pad_1, 0, pad_2), mode="reflect")
        HF_patch = F.pad(HF, (0, pad_1, 0, pad_2), mode="reflect")
        VF_patch = F.pad(VF, (0, pad_1, 0, pad_2), mode="reflect")
        DF_patch = F.pad(DF, (0, pad_1, 0, pad_2), mode="reflect")
        abun_get = self.backbone([LF_patch, HF_patch, VF_patch, DF_patch])
        abun_get = abun_get[:, :, :w, :h]
        recon_linear = torch.einsum('brhw,rl->blhw', [abun_get, endm_get])
        return recon_linear + 1e-7, abun_get, endm_get
    
    def get_endmember(self, lf_feature=None):
        
        query_split = torch.chunk(self.query_embed.weight, self.num_endm, dim=0)  
        query_split = torch.stack(query_split, dim=0)  
        
        
        weights_expand = self.weights.unsqueeze(-1).repeat(1, 1, self.num_band)  
        endm_init = query_split * weights_expand  
        
        
        if lf_feature is not None:  
            global_spe_feat = self.extract_global_spectral_feat(lf_feature) 
            
            endm_init_avg = torch.mean(endm_init, dim=1)
            gcs_matrix = self.compute_gcs_matrix(global_spe_feat, endm_init_avg)
            gcs_matrix_expand = gcs_matrix.unsqueeze(1).repeat(1, self.num_queries_times, 1)
            
            
            
            endm_weighted = endm_init #* (1 + self.gcs_scale * gcs_matrix_expand)
            
            #endm_weighted = endm_init 
        else:
            endm_weighted = endm_init
        #endm_weighted = endm_init
        
        endm_get = torch.mean(endm_weighted, dim=1)  
        endm_get = torch.clamp(endm_get, min=1e-7)  
        return endm_get

        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_endmember = 4
    num_band = 198
    rows = [100, 100]
    model = WMSM_Net(num_band).to(device)
    print(model)
    summary(model, [num_band, 100, 100])
    input_data = torch.randn(1, num_band, 100, 100).to(device)
    recon_linear, abun_get, endm_get = model(input_data)
    print(recon_linear.shape)
    print(abun_get.shape)
    print(endm_get.shape)

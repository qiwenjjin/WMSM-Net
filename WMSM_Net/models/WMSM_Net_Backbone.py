import math
import torch
from torch import nn
from mamba_ssm import Mamba
import torch.nn.functional as F
from einops import rearrange

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.spatial.distance import pdist, squareform
import numpy as np
import numpy as np
from scipy.spatial import ConvexHull
import math
from WMSM_Net_pooling import PatchEmbed, LayerNormChannel, GroupNorm, Mlp
from WMSM_Net_hybrid import WMSM_Net_HybridBlock
from functools import partial, reduce
from WMSM_Net_Feature import FMB

import torch
import torch.nn as nn
import numbers
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FDFA(nn.Module):
    def __init__(self, dim, num_heads, ):
        super(FDFA, self).__init__()
        self.num_heads = num_heads

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv1_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_1_2 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)


        self.conv2_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv2_1_2 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)




    def forward(self, x1,x2):
        b, c, h, w = x1.shape
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        attn_111 = self.conv1_1_1(x1)
        attn_112 = self.conv1_1_2(x1)

        attn_211 = self.conv2_1_1(x2)
        attn_212 = self.conv2_1_2(x2)


        out1 = attn_111 + attn_112
        out2 = attn_211 + attn_212
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
      
        k1 = rearrange(x1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(x1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(x2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(x2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        # Match each query to the key/value orientation of its attention path.
        # The previous axes only worked when h == w; Cuprite is rectangular.
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        


        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        out = self.project_out(out3) + self.project_out(out4) + x1+x2   

        return out



class Cross_atten(nn.Module):
    def __init__(self, dim, num_heads, ):
        super(Cross_atten, self).__init__()
        self.num_heads = num_heads

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv1_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv2_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv2_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)



    def forward(self, x1,x2):
        b, c, h, w = x1.shape
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        attn_111 = self.conv1_1_1(x1)
        attn_121 = self.conv1_2_1(x1)

        attn_211 = self.conv2_1_1(x2)
        attn_221 = self.conv2_2_1(x2)


        out1 = attn_111 + attn_121
        out2 = attn_211 + attn_221
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out3) + self.project_out(out4) + x1+x2

        return out

def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x


class CPCA(nn.Module):
    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

class SumToOne(nn.Module):
    def __init__(self, scale=3.5):
        super(SumToOne, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = torch.softmax(self.scale * x, dim=1)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale + x


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CA(nn.Module):
    def __init__(self, in_channels):
        super(CA, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.scale_param = nn.Parameter(torch.ones(in_channels))

    def forward(self, x):
        max_pooled = self.max(x)
        avg_pooled = self.avg(x)
        pooled_sum = max_pooled + avg_pooled
        scaled_pooled_sum = F.sigmoid(pooled_sum * self.scale_param.unsqueeze(-1).unsqueeze(-1))
        output = x * scaled_pooled_sum + x
        return output

class SpeMamba(nn.Module):
    def __init__(self, channels, use_residual=True, ds=4):
        super(SpeMamba, self).__init__()
        self.use_residual = use_residual
        self.down_ratio = ds
        self.ds = ds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.up_sample = nn.Upsample(scale_factor=ds, mode='bilinear')
        self.mamba = None

        self.proj = None

    def forward(self, x):
        
        if self.mamba is None or self.proj is None:
            self._build_model(x)
        x_unfolded = x.unfold(2, self.ds, self.ds).unfold(3, self.ds, self.ds)
        
        
        
        x_small = x_unfolded.mean(dim=(-1, -2))
        _, _, h, w = x_small.shape
        x_flat = rearrange(x_small, ' b c h w -> b c (h w)')
        x_flat = self.mamba(x_flat)
        x_proj = self.proj(x_flat)
        x_proj = rearrange(x_proj, ' b c (h w) -> b c h w', h=h)
        x_proj = self.up_sample(x_proj)

        if self.use_residual:
            return x + x_proj
        else:
            return x_proj

    def _build_model(self, x):
        
        batch_size, channels, height, width = x.shape
        self.d_model = int(height/self.down_ratio * width/self.down_ratio)
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=self.d_model,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        ).to(self.device)

        self.proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            #nn.LayerNorm(self.d_model),
            DyT(self.d_model),
            nn.SiLU(),
            nn.Softmax(dim=-1)
        ).to(self.device)


class SpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, use_proj=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=channels,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.Linear(channels, channels),
                #nn.LayerNorm(channels),
                DyT(channels),
                nn.SiLU(),
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_re.shape
        x_flat = x_re.view(1, -1, C)
        x_flat = self.mamba(x_flat)

        if self.use_proj:
            x_flat = self.proj(x_flat)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()

        if self.use_residual:
            return x_recon + x
        else:
            return x_recon

class BothMamba(nn.Module):
    def __init__(self, channels, use_residual, use_att=True, ds=4):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            # self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(channels, use_residual=use_residual)
        self.spe_mamba = SpeMamba(channels, use_residual=use_residual, ds=ds)
        self.ca = CA(channels)
        #self.CPCA = CPCA(channels, channels)
        
        #self.model = FFM(channels, channels)
        #self.abu = abu_similarity(data)
        
    def forward(self, x):
        ca_x1 = self.ca(x)
        #ca_x1 = self.CPCA(x)
        spa_x = self.spa_mamba(ca_x1)
        ca_x2 = self.ca(spa_x)
        #ca_x2 = self.CPCA(spa_x)
        spe_x = self.spe_mamba(ca_x2)
        if self.use_att:
            fusion_x = spa_x * self.weights[0] + spe_x * self.weights[1]
            #fusion_x = self.model(spa_x, spe_x)
        else:
            fusion_x = spa_x + spe_x
        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x


class MultiScaleSpaMamba(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale1 = SpaMamba(channels, use_residual=True)
        self.down2 = nn.AvgPool2d(2)
        self.scale2 = SpaMamba(channels, use_residual=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        x1 = self.scale1(x)
        x2 = self.down2(x1)
        x2 = self.scale2(x2)
        x2_up = self.up2(x2)
        
        if x2_up.shape[2:] != x1.shape[2:]:
            x2_up = F.interpolate(x2_up, size=x1.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([x1, x2_up], dim=1)
        out = self.fuse(fused)
        return out + x

class SpeMambaGroup(nn.Module):
    def __init__(self, channels, num_groups=2, ds=4):
        super().__init__()
        assert channels % num_groups == 0
        self.num_groups = num_groups
        self.groups = nn.ModuleList([
            SpeMamba(channels // num_groups, use_residual=True, ds=ds)
            for _ in range(num_groups)
        ])
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        xs = torch.chunk(x, self.num_groups, dim=1)
        outs = [layer(x_i) for x_i, layer in zip(xs, self.groups)]
        out = torch.cat(outs, dim=1)
        return self.fuse(out) + x

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba

class ConvMambaSpectralBlock(nn.Module):
    def __init__(self, num_bands, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        
        
        self.short_conv = nn.Sequential(
            
            nn.Conv2d(num_bands, num_bands, kernel_size=(5, 1), padding=(2, 0), groups=num_bands),
            nn.Conv2d(num_bands, num_bands, kernel_size=(1, 5), padding=(0, 2), groups=num_bands),
            nn.GELU()
        )
        
        
        self.long_mamba = Mamba(
            d_model=num_bands,   
            d_state=16,          
            d_conv=4,            
            expand=2             
        )
        
        
        self.proj = nn.Conv2d(num_bands, num_bands, kernel_size=1)

    def forward(self, x):
        """
        x: [B, C, H, W], C = num_bands
        """
        B, C, H, W = x.shape
        
        
        x_short = self.short_conv(x)  # [B, C, H, W]
        
        
        # reshape to [B, H*W, C] for Mamba
        x_seq = rearrange(x_short, 'b c h w -> b (h w) c')
        x_mamba = self.long_mamba(x_seq)  # [B, H*W, C]
        x_long = rearrange(x_mamba, 'b (h w) c -> b c h w', h=H, w=W)
        
        
        x_out = self.proj(x_long)
        
        if self.use_residual:
            return x + x_out
        else:
            return x_out

def conv33(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)

class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(2,1,7,1,3),
            nn.Sigmoid())
        self.processpha = nn.Sequential(
            nn.Conv2d(2, 1, 7, 1, 3),
            nn.Sigmoid())

        self.process1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True))

    def forward(self,x):
        _,_,H,W = x.shape
        # x = self.proj(x)
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        mag_avg = torch.mean(mag, dim=1, keepdim=True) 
        mag_max = torch.max(mag, dim=1, keepdim=True)[0]
        pha_avg = torch.mean(pha, dim=1, keepdim=True) 
        pha_max = torch.max(pha, dim=1, keepdim=True)[0]
        mag = mag * self.processmag(torch.cat([mag_avg, mag_max],dim=1))
        pha = pha * self.processpha(torch.cat([pha_avg, pha_max],dim=1))
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_freq_spatial = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        return x_freq_spatial

import torch
import torch.nn as nn
import torch.fft

class FFTInteraction_N(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spatial_proj = nn.Conv2d(dim, dim, 1)
        self.freq_proj = nn.Conv2d(dim, dim, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # spatial
        s = self.spatial_proj(x)
        # frequency
        f = torch.fft.irfft2(torch.fft.rfft2(x, norm='ortho'), s=x.shape[-2:], norm='ortho')
        f = self.freq_proj(f)
        # gate
        g = self.gate(torch.cat([s, f], dim=1))
        out = g * s + (1 - g) * f
        return out

class MBRConv1X1(nn.Module):
    def __init__(self,in_channels,out_channels,rep_scale=4):
        super(MBRConv1X1, self).__init__()
        self.in_channels=in_channels 
        self.out_channels=out_channels 
        self.rep_scale=rep_scale 
        
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, kernel_size=1)
        
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, kernel_size=1)
        self.conv_out.weight.requires_grad=False 
        
        self.weight1 = nn.Parameter(torch.zeros_like(self.conv_out.weight))
        nn.init.xavier_normal_(self.weight1)
    
    def forward(self,inp): 
        
        x1=self.conv(inp) 
        
        x=torch.cat([x1,self.conv_bn(x1)],dim=1)
        
        final_weight = self.conv_out.weight + self.weight1
        
        out = F.conv2d(x,final_weight, bias=self.conv_out.bias, stride=1, padding=0, dilation=1, groups = 1)
        return out 
        

class HDPA(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(HDPA, self).__init__()
        self.channels = channels
        
        self.globalatt = nn.Sequential(
            MBRConv1X1(channels, channels, rep_scale=rep_scale),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()  
        )
        
        self.localatt = nn.Sequential(
            MBRConv1X1(1, channels, rep_scale=rep_scale),
            nn.Sigmoid()  
        )
        
        self.local_conv = MBRConv1X1(
            in_channels=1,  
            out_channels=channels,  
            rep_scale=rep_scale
        )
        self.local_act = nn.Sigmoid()  


    def forward(self, x):

        x1 = self.globalatt(x) 
        
        max_out,_ = torch.max(x1 * x, dim = 1, keepdim = True)
        
        x2 = self.localatt(max_out)
        
        x3 = torch.mul(x1, x2) * x
        
        return x3

def conv33(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)  



    def forward(self, x, alpha):
        B, C_in, H, W = x.shape
        _, C_out, _, K, _ = self.weight.shape
        x = x.view(1, -1, H, W)
        weight = (alpha.unsqueeze(1) * self.weight).view(-1, C_in, K, K)
        bias = self.bias.repeat(B, 1).view(-1)
        out = F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding, groups=B)
        return out.view(B, C_out, out.size(-2), out.size(-1))

class SpecDynConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.b = SpecAdaConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.b_rf = RouteFuncMLP(c_in=in_channels)
    def forward(self, x):
        return self.b(x, self.b_rf(x))


def xcorr_fast(x, kernel, kwargs):
    batch = kernel.size(0)
    pk = kernel.view(-1, x.size(1), kernel.size(2), kernel.size(3))
    px = x.view(1, -1, x.size(2), x.size(3))
    po = F.conv2d(px, pk, **kwargs, groups=batch)
    return po.view(batch, -1, po.size(2), po.size(3))

class Correlation(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x, kernel, **kwargs):
        return xcorr_fast(x, kernel, kwargs)


class asign_index(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, guide_feature):
        ctx.save_for_backward(kernel, guide_feature)
        guide_mask = torch.zeros_like(guide_feature).scatter_(
            1, guide_feature.argmax(dim=1, keepdim=True), 1
        ).unsqueeze(2)
        return (kernel * guide_mask).sum(dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        kernel, guide_feature = ctx.saved_tensors
        guide_mask = torch.zeros_like(guide_feature).scatter_(
            1, guide_feature.argmax(dim=1, keepdim=True), 1
        ).unsqueeze(2)
        grad_kernel = grad_output.unsqueeze(1) * guide_mask
        grad_guide = (grad_output.unsqueeze(1) * kernel).sum(dim=2)
        softmax = F.softmax(guide_feature, 1)
        grad_guide = softmax * (grad_guide - (softmax * grad_guide).sum(dim=1, keepdim=True))
        return grad_kernel, grad_guide


class SSRA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, region_num=8):
        super(SSRA, self).__init__()
        self.region_num = region_num
        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, 1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, 1, groups=region_num)
        )
        self.corr = Correlation()
        self.spectral_guide = SpecClassifier(input_dim=in_channels, output_dim=region_num)
        self.spatial_guide = SpecDynConv(in_channels, region_num)
        self.fution_ss = nn.Conv2d(in_channels * 2, out_channels, 3, 1, 1, bias=True)
        self.asign_index = asign_index.apply

    def forward(self, input, guide_input=None):
        if guide_input is None:
            guide_input = input
        kernel = self.conv_kernel(input)
        output = self.corr(input, kernel)  # [B, R*out, H, W]
        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3))

        guide_spa = self.spatial_guide(guide_input)
        guide_spec = self.spectral_guide(guide_input)

        out_spa = self.asign_index(output, guide_spa)
        out_spec = self.asign_index(output, guide_spec)

        out_fused = self.fution_ss(torch.cat([out_spa, out_spec], dim=1))
        return out_fused + input
   
class MSC_BothMamba(nn.Module):
    def __init__(self, channels, use_residual=True, use_att=True, ds=4, rep_scale=4, dropout=5e-2):
        super().__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        
        self.spe_enh = FMB(channels, ffn_scale=2.0)  
        self.spa_enh = FMB(channels, ffn_scale=2.0)  
        
        
        #self.layer = nn.Sequential(
        
        #    nn.LeakyReLU(0.2),
        #    nn.BatchNorm2d(96),
        #    nn.Dropout(dropout),
        
        #    nn.LeakyReLU(0.2),
        #    nn.BatchNorm2d(48),
        #    nn.Dropout(dropout),
        
        #)
        
        
        #self.spa_mamba = SpaMamba(channels, use_residual=use_residual)
        self.spe_mamba = SpeMamba(channels, use_residual=use_residual, ds=ds)
        self.spa_mamba = MultiScaleSpaMamba(channels)
        self.ca = CA(channels)  
        self.hdpa = HDPA(channels)  
        
        self.frequency = FFTInteraction_N(dim = channels)
        #self.gate = nn.Sequential(
        #    nn.Conv2d(dim * 3, dim, 1, 1, 0),
        #    nn.GELU(),
        #    nn.Conv2d(dim, dim, 1, 1, 0),
        #)

    def forward(self, x):
        #ca_x1 = self.ca(x)
        #spe_x = self.spe_enh(x)
        spe_x = self.spe_mamba(x)
        
        #spe_x = self.spe_enh(spe_x)
        spa_x = self.spa_mamba(spe_x)
        
        #spa_x = self.spa_enh(spa_x)
        #ca_x2 = self.ca(spa_x)
    
        #en_x = self.frequency(spe_x)
        
        
        fusion_x = spa_x + spe_x #+ en_x
        
        fusion_x = self.hdpa(fusion_x)  
        fusion_x = self.ca(fusion_x)
         
        if self.use_residual:
            return fusion_x + x 
        else:
            return fusion_x 

import pywt  

class BSPSWT(nn.Module):
    def __init__(self, wavelet='bior3.3', level=1):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        
        self.wavelet_obj = pywt.Wavelet(wavelet)
    
    def forward(self, x):
        ""
        B, C, H, W = x.shape
        
        
        
        pad_h = 1 if H % 2 != 0 else 0
        pad_w = 1 if W % 2 != 0 else 0
        
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            
            H_padded, W_padded = H + pad_h, W + pad_w
        else:
            H_padded, W_padded = H, W
        
        
        lf_list, hf_list, vf_list, df_list = [], [], [], []
        for b in range(B):
            for c in range(C):
                
                img = x[b, c].cpu().numpy()  
                
                
                coeffs = pywt.swt2(img, self.wavelet_obj, level=self.level)[0]
                lf = coeffs[0]  
                hf, vf, df = coeffs[1]  
                
                
                lf = torch.tensor(lf, dtype=x.dtype).to(x.device).unsqueeze(0).unsqueeze(0)
                hf = torch.tensor(hf, dtype=x.dtype).to(x.device).unsqueeze(0).unsqueeze(0)
                vf = torch.tensor(vf, dtype=x.dtype).to(x.device).unsqueeze(0).unsqueeze(0)
                df = torch.tensor(df, dtype=x.dtype).to(x.device).unsqueeze(0).unsqueeze(0)
                
                lf_list.append(lf)
                hf_list.append(hf)
                vf_list.append(vf)
                df_list.append(df)
        
        
        LF = torch.cat(lf_list, dim=1).reshape(B, C, H_padded, W_padded)
        HF = torch.cat(hf_list, dim=1).reshape(B, C, H_padded, W_padded)
        VF = torch.cat(vf_list, dim=1).reshape(B, C, H_padded, W_padded)
        DF = torch.cat(df_list, dim=1).reshape(B, C, H_padded, W_padded)
        
        
        if pad_h > 0 or pad_w > 0:
            LF = LF[..., :H, :W]  
            HF = HF[..., :H, :W]
            VF = VF[..., :H, :W]
            DF = DF[..., :H, :W]
        
        return LF, HF, VF, DF

class BSPDWT(nn.Module):
    def __init__(self, wavelet='bior3.3', level=1):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.wavelet_obj = pywt.Wavelet(wavelet)
    
    def forward(self, x):
        ""
        B, C, H, W = x.shape
        
        
        pad_h = 1 if H % 2 != 0 else 0
        pad_w = 1 if W % 2 != 0 else 0
        
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_padded = x
            
        
        lf_list, hf_list, vf_list, df_list = [], [], [], []
        
        
        x_np = x_padded.cpu().detach().numpy()
        
        for b in range(B):
            for c in range(C):
                img = x_np[b, c]
                
                
                # data: (cA, (cH, cV, cD))
                coeffs = pywt.dwt2(img, self.wavelet_obj)
                lf, (hf, vf, df) = coeffs
                
                
                
                lf = torch.from_numpy(lf).to(x.device).unsqueeze(0).unsqueeze(0)
                hf = torch.from_numpy(hf).to(x.device).unsqueeze(0).unsqueeze(0)
                vf = torch.from_numpy(vf).to(x.device).unsqueeze(0).unsqueeze(0)
                df = torch.from_numpy(df).to(x.device).unsqueeze(0).unsqueeze(0)
                
                lf_list.append(lf)
                hf_list.append(hf)
                vf_list.append(vf)
                df_list.append(df)
        
        
        
        def stack_and_reshape(t_list):
            t = torch.cat(t_list, dim=0) # [B*C, 1, h, w]
            
            
            return t.view(B, C, t.shape[2], t.shape[3])

        LF_small = stack_and_reshape(lf_list)
        HF_small = stack_and_reshape(hf_list)
        VF_small = stack_and_reshape(vf_list)
        DF_small = stack_and_reshape(df_list)
        
        
        
        
        
        target_size = (x_padded.shape[2], x_padded.shape[3])
        
        LF = F.interpolate(LF_small, size=target_size, mode='bilinear', align_corners=False)
        HF = F.interpolate(HF_small, size=target_size, mode='bilinear', align_corners=False)
        VF = F.interpolate(VF_small, size=target_size, mode='bilinear', align_corners=False)
        DF = F.interpolate(DF_small, size=target_size, mode='bilinear', align_corners=False)
        
        
        if pad_h > 0 or pad_w > 0:
            LF = LF[..., :H, :W]
            HF = HF[..., :H, :W]
            VF = VF[..., :H, :W]
            DF = DF[..., :H, :W]
            
        return LF, HF, VF, DF

class MSC_WMSM_NetBackbone(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, 
                 group_num=4, scale=3.5, ds=4, dropout=5e-2, num_heads=8):
        super().__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.group_num = group_num 
        self.num_heads = num_heads 
        
        
        self.patch_embedding = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1),
                nn.GroupNorm(group_num, hidden_dim),
                nn.SiLU()
            ) for _ in range(4)  
        ])
          
        
        
        self.lf_branch = nn.Sequential(
            WMSM_Net_HybridBlock(
                dim=hidden_dim,
                token_mixer=partial(MSC_BothMamba, ds=ds, dropout=dropout),  
                mlp_ratio=4.,
                norm_layer=GroupNorm,
                act_layer=nn.GELU,
                drop=0.,
                drop_path=0.,
                use_layer_scale=True,
                layer_scale_init_value=1e-5,
            ),
            WMSM_Net_HybridBlock(
                dim=hidden_dim,
                token_mixer=partial(MSC_BothMamba, ds=ds, dropout=dropout),  
                mlp_ratio=4.,
                norm_layer=GroupNorm,
                act_layer=nn.GELU,
                drop=0.,
                drop_path=0.,
                use_layer_scale=True,
                layer_scale_init_value=1e-5,
            ),
            WMSM_Net_HybridBlock(
                dim=hidden_dim,
                token_mixer=partial(MSC_BothMamba, ds=ds, dropout=dropout),  
                mlp_ratio=4.,
                norm_layer=GroupNorm,
                act_layer=nn.GELU,
                drop=0.,
                drop_path=0.,
                use_layer_scale=True,
                layer_scale_init_value=1e-5,
            ),
            


        )
        
        self.hf_branch = nn.Sequential(
            
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 5), padding=(0, 2), groups=hidden_dim),
            WMSM_Net_HybridBlock(
                dim=hidden_dim,
                token_mixer=partial(MSC_BothMamba, ds=ds, dropout=dropout),  
                mlp_ratio=4.,
                norm_layer=GroupNorm,
                act_layer=nn.GELU,
                drop=0.,
                drop_path=0.,
                use_layer_scale=True,
                layer_scale_init_value=1e-5,
            ),
            WMSM_Net_HybridBlock(
                dim=hidden_dim,
                token_mixer=partial(MSC_BothMamba, ds=ds, dropout=dropout),  
                mlp_ratio=4.,
                norm_layer=GroupNorm,
                act_layer=nn.GELU,
                drop=0.,
                drop_path=0.,
                use_layer_scale=True,
                layer_scale_init_value=1e-5,
            ),
            WMSM_Net_HybridBlock(
                dim=hidden_dim,
                token_mixer=partial(MSC_BothMamba, ds=ds, dropout=dropout),  
                mlp_ratio=4.,
                norm_layer=GroupNorm,
                act_layer=nn.GELU,
                drop=0.,
                drop_path=0.,
                use_layer_scale=True,
                layer_scale_init_value=1e-5,
            ),


  
        )

        self.vf_branch = nn.Sequential(
            
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 1), padding=(2, 0), groups=hidden_dim),
            WMSM_Net_HybridBlock(
                dim=hidden_dim,
                token_mixer=partial(MSC_BothMamba, ds=ds, dropout=dropout),  
                mlp_ratio=4.,
                norm_layer=GroupNorm,
                act_layer=nn.GELU,
                drop=0.,
                drop_path=0.,
                use_layer_scale=True,
                layer_scale_init_value=1e-5,
            ),
            WMSM_Net_HybridBlock(
                dim=hidden_dim,
                token_mixer=partial(MSC_BothMamba, ds=ds, dropout=dropout),  
                mlp_ratio=4.,
                norm_layer=GroupNorm,
                act_layer=nn.GELU,
                drop=0.,
                drop_path=0.,
                use_layer_scale=True,
                layer_scale_init_value=1e-5,
            ),
            WMSM_Net_HybridBlock(
                dim=hidden_dim,
                token_mixer=partial(MSC_BothMamba, ds=ds, dropout=dropout),  
                mlp_ratio=4.,
                norm_layer=GroupNorm,
                act_layer=nn.GELU,
                drop=0.,
                drop_path=0.,
                use_layer_scale=True,
                layer_scale_init_value=1e-5,
            ),



        )

        self.df_branch = nn.Sequential(
            
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            WMSM_Net_HybridBlock(
                dim=hidden_dim,
                token_mixer=partial(MSC_BothMamba, ds=ds, dropout=dropout),  
                mlp_ratio=4.,
                norm_layer=GroupNorm,
                act_layer=nn.GELU,
                drop=0.,
                drop_path=0.,
                use_layer_scale=True,
                layer_scale_init_value=1e-5,
            ),
            WMSM_Net_HybridBlock(
                dim=hidden_dim,
                token_mixer=partial(MSC_BothMamba, ds=ds, dropout=dropout),  
                mlp_ratio=4.,
                norm_layer=GroupNorm,
                act_layer=nn.GELU,
                drop=0.,
                drop_path=0.,
                use_layer_scale=True,
                layer_scale_init_value=1e-5,
            ),
            WMSM_Net_HybridBlock(
                dim=hidden_dim,
                token_mixer=partial(MSC_BothMamba, ds=ds, dropout=dropout),  
                mlp_ratio=4.,
                norm_layer=GroupNorm,
                act_layer=nn.GELU,
                drop=0.,
                drop_path=0.,
                use_layer_scale=True,
                layer_scale_init_value=1e-5,
            ),



        )
            
        
        self.branch_weights = nn.Parameter(torch.ones(4))  
        
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=1),
            nn.GroupNorm(group_num * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)  
        )
        
        
        
        
        self.fdfa_hf = FDFA(dim=hidden_dim, num_heads=num_heads) # LF <-> HF
        self.fdfa_vf = FDFA(dim=hidden_dim, num_heads=num_heads) # LF <-> VF
        self.fdfa_df = FDFA(dim=hidden_dim, num_heads=num_heads) # LF <-> DF

        
        
        
        self.final_fuse = nn.Sequential(
            #nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=1),
            nn.Conv2d(hidden_dim * 3, hidden_dim * 2, kernel_size=1),
            nn.GroupNorm(group_num * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1) 
        )
        
        
        self.abundance_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 1),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(128, num_classes, 1),
            nn.BatchNorm2d(num_classes),
            SumToOne(scale=scale),
        )

    def forward(self, x_list):
        ""
        
        lf_out, hf_out, vf_out, df_out = [
            self.patch_embedding[i](x) for i, x in enumerate(x_list)
        ]
        
        
        
        lf_out = self.lf_branch(lf_out)
        
        hf_out = self.hf_branch(hf_out)
        vf_out = self.vf_branch(vf_out)
        df_out = self.df_branch(df_out)
        
        
               
        
        fused = torch.cat([lf_out, hf_out, vf_out, df_out], dim=1)  # [B, 4*hidden_dim, H, W]
        fused = self.fuse(fused)  # [B, hidden_dim, H, W]

        
        
        
        
        #weighted_fused = (
        #    lf_out * weights[0] +  
        #    hf_out * weights[1] +  
        #    vf_out * weights[2] +  
        #    df_out * weights[3]    
        #)
        
        
        
        #fused = torch.cat([
        #    lf_out * weights[0],
        #    hf_out * weights[1], 
        #    vf_out * weights[2],
        #    df_out * weights[3]
        #], dim=1)
        #fused = self.fuse(fused)

        
        weights = F.softmax(self.branch_weights, dim=0) # [w_lf, w_hf, w_vf, w_df]
        
        
        x_LF_w = lf_out * weights[0]
        x_HF_w = hf_out * weights[1]
        x_VF_w = vf_out * weights[2]
        x_DF_w = df_out * weights[3]
        
        
        
        # LF <-> HF
        
        fused_lf_hf = self.fdfa_hf(x_LF_w, x_HF_w) # [B, hidden_dim, H, W]
        
        # LF <-> VF
        fused_lf_vf = self.fdfa_vf(x_LF_w, x_VF_w) # [B, hidden_dim, H, W]
        
        # LF <-> DF
        fused_lf_df = self.fdfa_df(x_LF_w, x_DF_w) # [B, hidden_dim, H, W]
        
        
        fused_cat = torch.cat([fused_lf_hf, fused_lf_vf, fused_lf_df], dim=1) # [B, 3*hidden_dim, H, W]
        #fused_cat = torch.cat([x_LF_w, fused_lf_hf, fused_lf_vf, fused_lf_df], dim=1) # [B, 3*hidden_dim, H, W]
        
        fused = self.final_fuse(fused_cat) # [B, hidden_dim, H, W]



        
        abun = self.abundance_head(fused)
        #abun = self.abundance_head(lf_out)

        if self.training:
            abun = F.dropout2d(abun, p=self.dropout)
        return abun 

                        
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, length, dim = 2, 512 * 512, 256
    x = torch.randn(batch, length, dim).to("cuda")
    print(x.shape)
    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim,  # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,  # Local convolution width
        expand=2,  # Block expansion factor
    ).to("cuda")
    y = model(x)
    print(y.shape)

    num_endmember = 3
    num_band = 173
    rows = [100, 100]
    # rows = 100
    model = WMSM_NetBackbone(in_channels=num_band, num_classes=num_endmember).to(device)
    print(model)
    # summary(model, [num_band, 100, 100])
    input_data = torch.randn(1, num_band, rows[0], rows[1]).to(device)
    pred_abun = model(input_data)
    print(pred_abun.shape)

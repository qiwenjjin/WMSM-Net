'''
@author: Chen Dong
HIT

'''
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import random
import scipy.io as sio
from torch.utils.data import DataLoader
from utils import plots
import argparse
from utils.utils import *
from utils.data_read import data_get_func
from utils.metrics_extra import extra_metrics
from models.WMSM_Net import WMSM_Net
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment  


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def seed_torch(seed=1):
    '''
    Keep the seed fixed thus the results can keep stable
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def match_endmembers(est_endm, gt_endm):
    ""
    
    est_norm_val = np.linalg.norm(est_endm, axis=0, keepdims=True)
    gt_norm_val = np.linalg.norm(gt_endm, axis=0, keepdims=True)
    
    
    norm_est = est_endm / (est_norm_val + 1e-8)
    norm_gt = gt_endm / (gt_norm_val + 1e-8)
    
    
    cos_sim = np.dot(norm_est.T, norm_gt)
    
    
    if np.isnan(cos_sim).any():
        print("Warning: NaN detected in cosine similarity matrix. Replacing with -1.")
        
        cos_sim = np.nan_to_num(cos_sim, nan=-1.0)
    
    
    row_ind, col_ind = linear_sum_assignment(-cos_sim)  
    return col_ind


def compute_rmse(x_true, x_pre):
    w, h, c = x_true.shape
    class_rmse = [0] * c
    for i in range(c):
        class_rmse[i] = np.sqrt(((x_true[:, :, i] - x_pre[:, :, i]) ** 2).sum() / (w * h))
    mean_rmse = np.sqrt(((x_true - x_pre) ** 2).sum() / (w * h * c))
    return class_rmse, mean_rmse


def compute_sad(inp, target):
    p = inp.shape[-1]
    sad_err = [0] * p
    for i in range(p):
        inp_norm = np.linalg.norm(inp[:, i], 2)
        tar_norm = np.linalg.norm(target[:, i], 2)
        summation = np.matmul(inp[:, i].T, target[:, i])
        
        summation = np.clip(summation / (inp_norm * tar_norm + 1e-8), -1.0, 1.0)
        sad_err[i] = np.arccos(summation)
    mean_sad = np.mean(sad_err)
    return sad_err, mean_sad
    

def evaluate_vca_quality(data_get, save_path):
    ""
    
    hsi_img = data_get.data_hsi.reshape(data_get.num_bands, -1)  # [L, N] N=H*W
    gt_endmembers = data_get.data_endm  # [L, R]
    vca_endmembers = data_get.VCA_endmember  # [L, R]
    gt_abundances = data_get.data_aban.reshape(data_get.num_endm, -1)  # [R, N]
    
    
    vca_match_idx = match_endmembers(vca_endmembers, gt_endmembers)
    vca_endmembers_matched = vca_endmembers[:, vca_match_idx]
    
    
    vca_sad_per_endm, vca_mean_sad = compute_sad(vca_endmembers_matched, gt_endmembers)
    
    
    print("\n==================== VCA Endmember Quality Evaluation ====================")
    print("=== VCA Endmembers vs Ground Truth Endmembers ===")
    for i, sad in enumerate(vca_sad_per_endm):
        print(f"Endmember {i}: SAD = {sad:.6f} rad ({np.degrees(sad):.2f}°)")
    print(f"Mean SAD: {vca_mean_sad:.6f} rad ({np.degrees(vca_mean_sad):.2f}°)")
    
    
    
    plot_vca_gt_comparison(gt_endmembers, vca_endmembers_matched, 
                          data_get.num_endm, save_path)
    
    return {
        'vca_sad_per_endm': vca_sad_per_endm,
        'vca_mean_sad': vca_mean_sad
    }


def plot_vca_gt_comparison(gt_endm, vca_endm, num_endm, save_path):
    ""
    plt.figure(figsize=(15, 5*num_endm))
    
    for i in range(num_endm):
        plt.subplot(num_endm, 1, i+1)
        plt.plot(gt_endm[:, i], 'b-', linewidth=2, label='Ground Truth')
        plt.plot(vca_endm[:, i], 'r--', linewidth=2, label='VCA Extracted')
        plt.title(f'Endmember {i+1} Comparison')
        plt.xlabel('Band Index')
        plt.ylabel('Reflectance')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'vca_vs_gt_endmembers.png'), dpi=300, bbox_inches='tight')
    plt.close()
        

# ---------------defined paramaters------------------ #
parser = argparse.ArgumentParser(description='Training for HSI Unmixing')
parser.add_argument('-d', '--dataset', dest='dataset', choices=['S1', 'JR', 'UR', 'UR5', 'UR6', 'SA', 'AP'],
                    default='JR', help="Name of dataset.")
parser.add_argument('-i', '--iter', type=int, dest='iter', default=1, help="No of Monte Carlo test")
parser.add_argument('-e', '--epochs', type=int, default=300, help='epoch number')
parser.add_argument('-ds', '--down_ratio', type=int, default=16, help='down ratio')
parser.add_argument('-nq', '--num_queries', type=int, default=50, help='query number')
parser.add_argument('-b', '--batch_size', type=int, default=1, help='number of batch size')
parser.add_argument('-n', '--num_copies', type=int, default=1, help='number of copies')
parser.add_argument('--model_name', type=str, default='WMSM_Net', help='model used')
parser.add_argument('--seed', type=int, default=11, help='number of seed')

parser.add_argument('-l', '--lr', type=float, default=3e-3, help='learning rate')
parser.add_argument('--weight_mse', type=float, default=1, help='weight_mse')
parser.add_argument('--weight_sad', type=float, default=1, help='weight_sad')
parser.add_argument('--weight_sid', type=float, default=0, help='weight_sid')
parser.add_argument('--weight_endm', type=float, default=1, help='weight_endm')
parser.add_argument('--weight_aban', type=float, default=1e-2, help='weight_endm')
parser.add_argument('--dropout', type=float, default=0.03, help='weight_endm')
parser.add_argument('--weight_vca', type=float, default=0, help='weight of vca constraint') 
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
args = parser.parse_args()


seed_torch(seed=args.seed)
# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create saving path
dataset_query = {"AP": "Apex", "JR": "Jasper Ridge", "SA": "Samson", "UR": "Urban", "UR5": "Urban5", "UR6": "Urban6"}
workspace = os.path.abspath(".")
file_path = os.path.join(workspace, 'dataset', dataset_query[args.dataset])
save_path = os.path.join('results', args.model_name, dataset_query[args.dataset])

if not os.path.exists(save_path):
    os.makedirs(save_path)

model_save_path = os.path.join(save_path, 'model')
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

fig_save_path = os.path.join(save_path, 'index_fig')
if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)


with open(os.path.join(save_path, 'argparser_params.txt'), 'w') as f:
    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}\n")


data_get = data_get_func(dataname=dataset_query[args.dataset], file_path=file_path)
data_hsi_img = data_get.data_hsi.reshape(data_get.num_bands, data_get.num_rows, data_get.num_cols).transpose((0, 2, 1))
data_aban_img = data_get.data_aban.reshape(data_get.num_endm, data_get.num_rows, data_get.num_cols).transpose((0, 2, 1))
print(f"Data loaded, HSI shape: {data_hsi_img.shape}")


torch_hsi = torch.from_numpy(data_hsi_img).to(torch.float32).to(device).unsqueeze(0).repeat(args.num_copies, 1, 1, 1)
torch_aban = torch.from_numpy(data_aban_img).to(torch.float32).to(device)
hsi_mean_tensor = torch.from_numpy(data_get.get_hsi_mean()).to(torch.float32).to(device)

train_data_loader = DataLoader(torch_hsi, batch_size=args.batch_size, shuffle=False)


vca_endm_tensor = None
if hasattr(data_get, 'VCA_endmember'):
    
    vca_endm_tensor = torch.from_numpy(data_get.VCA_endmember).to(torch.float32).to(device)
    print("VCA Endmember Tensor loaded for loss constraint.")


model = WMSM_Net(
    num_band=data_get.num_bands, 
    d_model=64,
    num_endm=data_get.num_endm, 
    num_queries_times=args.num_queries, 
    ds=args.down_ratio,
    dropout=args.dropout,
    
    vca_endmember= None 
).to(device)

my_loss = My_Loss(
    weight_mse=args.weight_mse,
    weight_sad=args.weight_sad,
    weight_endm=args.weight_endm,
    weight_vca=args.weight_vca  
)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.9)

model.train()


test_hsi_img = data_hsi_img.copy()
test_aban_img = data_aban_img.copy().transpose((2, 1, 0))  
test_endm = data_get.data_endm.copy()  
torch_test_hsi = torch.from_numpy(test_hsi_img).to(torch.float32).to(device).unsqueeze(0)

for epoch in range(0, args.epochs):
    loss_print = 0
    temp_i = 0
    for i, x in enumerate(train_data_loader):
        temp_i += 1
        pred_linear, pred_abun, pred_endm = model(x)
        loss = my_loss(x, pred_linear, pred_endm, hsi_mean_tensor, pred_aban=pred_abun, vca_endm=vca_endm_tensor)
        optimizer.zero_grad()
        loss.backward()
        
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)        
                        
        for p in model.query_embed.weight:
            p.data.clamp_(1e-7, 1)
        optimizer.step()
        loss_print += loss
    loss_print = loss_print / (temp_i + 1)

    
    if (epoch + 1) % 5 == 0:
        model.eval()  
        with torch.no_grad():  
            
            re_linear, abu_est, endm_ = model(torch_test_hsi)
            
            
            abu_est = abu_est.squeeze(0).permute(2, 1, 0).detach().cpu().numpy()  # (H, W, num_endm)
            est_endmem = model.get_endmember().detach().cpu().numpy().T  
            
            
            #index = match_endmembers(est_endmem, test_endm)
            #index = [0, 2, 3, 1] #1
            #index = [0, 2, 1, 3] #2
            index = [3, 2, 1, 0] #3
            #index = [3, 2, 1, 0] #4
            #index = [1, 2, 0, 3] #5
            #index = [0, 2, 3, 1] 
            abu_est[:, :, np.arange(data_get.num_endm)] = abu_est[:, :, index]
            est_endmem[:, np.arange(data_get.num_endm)] = est_endmem[:, index]
            
            
            gt_abun = test_aban_img.transpose(1, 0, 2)  
            pred_abun = abu_est.transpose(1, 0, 2)
            rmse_list, mean_rmse = compute_rmse(gt_abun, pred_abun)
            
            
            sad_list, mean_sad = compute_sad(est_endmem, test_endm)

            
            re_linear_np = re_linear.squeeze(0).detach().cpu().numpy()
            extra = extra_metrics(est_endmem, test_endm, gt_abun, pred_abun,
                                  hsi_true=test_hsi_img, hsi_rec=re_linear_np)
            mean_sid, mean_aad, mean_aid, mean_re = (
                extra['SID'], extra['AAD'], extra['AID'], extra['RE'])
        print("\nEpoch: %d/%d" % (epoch + 1, args.epochs))
        
        print("=== RMSE (per class) ===")
        for i, rmse in enumerate(rmse_list):
            print(f"Class {i}: RMSE = {rmse:.4f}")
        print(f"Mean RMSE: {mean_rmse:.4f}")

        
        print("=== SAD (per endmember) ===")
        for i, sad in enumerate(sad_list):
            print(f"Endmember {i}: SAD = {sad:.6f} rad ({np.degrees(sad):.2f}°)")
        print(f"Mean SAD: {mean_sad:.6f} rad ({np.degrees(mean_sad):.2f}°)")

        
        print(f"Mean SID: {mean_sid:.6f} | Mean AAD: {mean_aad:.6f} | "
              f"Mean AID: {mean_aid:.6f} | RE: {mean_re:.6f}")

        model.train()  

    scheduler.step()


model.eval()
torch.save(model.state_dict(), os.path.join(model_save_path, 'model.pth'))


plots.plot_abundance(test_aban_img.transpose((1, 0, 2)), abu_est.transpose((1, 0, 2)),
                     data_get.num_endm,
                     save_dir=fig_save_path)
plots.plot_endmembers(data_get.data_endm, est_endmem, data_get.num_endm,
                      save_dir=fig_save_path, wavelength_nm=data_get.wavelength_nm)


print("\n==================== Final Metrics ====================")
print("=== RMSE (per class) ===")
for i, rmse in enumerate(rmse_list):
    print(f"Class {i}: RMSE = {rmse:.4f}")
print(f"Mean RMSE: {mean_rmse:.4f}")

print("\n=== SAD (per endmember) ===")
for i, sad in enumerate(sad_list):
    print(f"Endmember {i}: SAD = {sad:.6f} rad ({np.degrees(sad):.2f}°)")
print(f"Mean SAD: {mean_sad:.6f} rad ({np.degrees(mean_sad):.2f}°)")

print("\n=== Additional Metrics ===")
print(f"Mean SID: {mean_sid:.6f}")
print(f"Mean AAD: {mean_aad:.6f} rad ({np.degrees(mean_aad):.2f}°)")
print(f"Mean AID: {mean_aid:.6f}")
print(f"RE: {mean_re:.6f}")

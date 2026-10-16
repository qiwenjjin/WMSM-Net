'''
@author: Chen Dong
HIT
Modified to include SNR noise injection and .mat file saving.
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
import time

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


def add_gaussian_noise(data, snr_db):
    ""
    bands, rows, cols = data.shape
    
    
    signal_power = np.sum(data ** 2) / data.size
    
    
    # => P_noise = P_signal / 10^(SNR/10)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    
    
    sigma = np.sqrt(noise_power)
    
    
    noise = np.random.normal(0, sigma, data.shape)
    
    print(f"--- Noise Injection ---")
    print(f"Target SNR: {snr_db} dB")
    print(f"Signal Power: {signal_power:.6f}")
    print(f"Noise Power: {noise_power:.6f}")
    
    return data + noise

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def match_endmembers(est_endm, gt_endm):
    ""
    
    
    
    est_endm_T = est_endm.T
    gt_endm_T = gt_endm.T
    
    # Check shape
    if est_endm.shape[0] != gt_endm.shape[0]:
        raise ValueError(f"Endmember counts do not match: Est {est_endm.shape} vs GT {gt_endm.shape}")

    
    # Cosine Distance = 1 - Cosine Similarity
    
    # matrix shape: [R_est, R_gt]
    cosine_dist = cdist(est_endm_T, gt_endm_T, metric='cosine')
    
    
    # SAD = arccos(Cosine Similarity) = arccos(1 - Cosine Distance)
    
    cos_sim = 1.0 - cosine_dist
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    sad_matrix = np.arccos(cos_sim)
    
    
    
    
    est_indices, gt_indices = linear_sum_assignment(sad_matrix)
    
    
    
    
    
    
    num_gt = gt_endm.shape[1]
    reorder_idx = np.zeros(num_gt, dtype=int)
    
    
    
    
    
    # Position 0 (GT0): Should take Est1
    # Position 1 (GT1): Should take Est2
    # Position 2 (GT2): Should take Est0
    
    
    reorder_idx[gt_indices] = est_indices
    
    
    matched_sad = sad_matrix[est_indices, gt_indices].mean()
    # print(f"Matched Mean SAD: {matched_sad:.6f}")
    
    return reorder_idx


def compute_rmse(x_true, x_pre):
    w, h, c = x_true.shape
    class_rmse = [0] * c
    for i in range(c):
        class_rmse[i] = np.sqrt(((x_true[:, :, i] - x_pre[:, :, i]) ** 2).sum() / (w * h))
    mean_rmse = np.sqrt(((x_true - x_pre) ** 2).sum() / (w * h * c))
    return class_rmse, mean_rmse


def compute_re(x_true, x_pre):
    if isinstance(x_true, torch.Tensor):
        x_true = x_true.detach().cpu().numpy()
    if isinstance(x_pre, torch.Tensor):
        x_pre = x_pre.detach().cpu().numpy()

    if x_true.ndim == 3:
        bands, h, w = x_true.shape
        x_true = x_true.reshape(bands, -1)
        x_pre = x_pre.reshape(bands, -1)
    
    numerator = np.sum(x_true * x_pre, axis=0)
    norm_true = np.linalg.norm(x_true, axis=0)
    norm_pre = np.linalg.norm(x_pre, axis=0)
    
    cosine = numerator / (norm_true * norm_pre + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_error = np.arccos(cosine)
    mean_re = np.mean(angle_error)
    
    return mean_re


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
    hsi_img = data_get.data_hsi.reshape(data_get.num_bands, -1)
    gt_endmembers = data_get.data_endm
    vca_endmembers = data_get.VCA_endmember
    
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
parser.add_argument('-d', '--dataset', dest='dataset', choices=['S1', 'JR', 'UR', 'UR5', 'UR6', 'SA', 'AP', 'SYN'],
                    default='SYN', help="Name of dataset.")
parser.add_argument('-i', '--iter', type=int, dest='iter', default=1, help="No of Monte Carlo test")
parser.add_argument('-e', '--epochs', type=int, default=800, help='epoch number')
parser.add_argument('-ds', '--down_ratio', type=int, default=16, help='down ratio')
parser.add_argument('-nq', '--num_queries', type=int, default=50, help='query number')
parser.add_argument('-b', '--batch_size', type=int, default=1, help='number of batch size')
parser.add_argument('-n', '--num_copies', type=int, default=1, help='number of copies')
parser.add_argument('--model_name', type=str, default='WMSM_Net', help='model used')
parser.add_argument('--seed', type=int, default=11, help='number of seed')


parser.add_argument('--snr', type=float, default=30, help='SNR in dB (e.g., 10, 20, 30). If None, use original data.')

parser.add_argument('-l', '--lr', type=float, default=1e-2, help='learning rate')
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_query = {"AP": "Apex", "JR": "Japser Ridge", "SA": "Samson", "UR": "Urban", "UR5": "Urban5", "UR6": "Urban6", "SYN": "syn"}
workspace = os.path.abspath(".")
file_path = os.path.join(workspace, 'dataset', dataset_query[args.dataset])


snr_suffix = f"_SNR_{int(args.snr)}" if args.snr is not None else ""
save_path = os.path.join('results', args.model_name, dataset_query[args.dataset] + snr_suffix)

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


gt_hsi_img_clean = data_hsi_img.copy()


if args.snr is not None:
    print(f"\n[INFO] Injecting noise with SNR = {args.snr} dB")
    data_hsi_img = add_gaussian_noise(data_hsi_img, args.snr)
    
else:
    print("\n[INFO] No noise injected (Clean data).")


if hasattr(data_get, 'VCA_endmember'):
    vca_evaluation_results = evaluate_vca_quality(data_get, save_path)
else:
    print("\nWarning: No VCA endmembers found in data_get. Skipping VCA evaluation.")


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
    vca_endmember=data_get.VCA_endmember if hasattr(data_get, 'VCA_endmember') else None
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


tic = time.time()
print("\nStart Training...")

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

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            re_linear, abu_est, endm_ = model(torch_test_hsi)
            
            
            
            re_linear_np = re_linear.squeeze(0).detach().cpu().numpy()
            re_val = compute_re(gt_hsi_img_clean, re_linear_np)

            abu_est = abu_est.squeeze(0).permute(2, 1, 0).detach().cpu().numpy()
            est_endmem = model.get_endmember().detach().cpu().numpy().T
            
            index = match_endmembers(est_endmem, test_endm)
            abu_est[:, :, np.arange(data_get.num_endm)] = abu_est[:, :, index]
            est_endmem[:, np.arange(data_get.num_endm)] = est_endmem[:, index]
            
            gt_abun = test_aban_img.transpose(1, 0, 2)
            pred_abun = abu_est.transpose(1, 0, 2)
            rmse_list, mean_rmse = compute_rmse(gt_abun, pred_abun)
            
            sad_list, mean_sad = compute_sad(est_endmem, test_endm)

            
            extra = extra_metrics(est_endmem, test_endm, gt_abun, pred_abun)
            mean_sid, mean_aad, mean_aid = extra['SID'], extra['AAD'], extra['AID']

        print("\nEpoch: %d/%d" % (epoch + 1, args.epochs))
        print(f"=== RE (vs Clean GT) ===\nRE: {re_val:.6f}")

        print("=== RMSE (per class) ===")
        for i, rmse in enumerate(rmse_list):
            print(f"Class {i}: RMSE = {rmse:.4f}")
        print(f"Mean RMSE: {mean_rmse:.4f}")

        print("=== SAD (per endmember) ===")
        for i, sad in enumerate(sad_list):
            print(f"Endmember {i}: SAD = {sad:.6f} rad ({np.degrees(sad):.2f}°)")
        print(f"Mean SAD: {mean_sad:.6f} rad ({np.degrees(mean_sad):.2f}°)")

        
        print(f"Mean SID: {mean_sid:.6f} | Mean AAD: {mean_aad:.6f} | "
              f"Mean AID: {mean_aid:.6f}")

        model.train()

    scheduler.step()


toc = time.time()
elapsed_time = toc - tic

model.eval()
torch.save(model.state_dict(), os.path.join(model_save_path, 'model.pth'))

plots.plot_abundance(test_aban_img.transpose((1, 0, 2)), abu_est.transpose((1, 0, 2)),
                     data_get.num_endm,
                     save_dir=fig_save_path)
plots.plot_endmembers(data_get.data_endm, est_endmem, data_get.num_endm,
                      save_dir=fig_save_path, wavelength_nm=data_get.wavelength_nm)

print("\n==================== Final Metrics ====================")
print(f"Total Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"Final RE: {re_val:.6f}")
print(f"Mean RMSE: {mean_rmse:.4f}")
print(f"Mean SAD: {mean_sad:.6f} rad ({np.degrees(mean_sad):.2f}°)")
print(f"Mean SID: {mean_sid:.6f}")
print(f"Mean AAD: {mean_aad:.6f} rad ({np.degrees(mean_aad):.2f}°)")
print(f"Mean AID: {mean_aid:.6f}")


mat_filename = f'results_{args.dataset}_SNR_{args.snr if args.snr is not None else "Clean"}.mat'
mat_save_path = os.path.join(save_path, mat_filename)

print(f"\nSaving results to {mat_save_path}...")


mat_data = {
    'SAD_mean': mean_sad,
    'SAD_per_endmember': sad_list,
    'RMSE_mean': mean_rmse,
    'RMSE_per_class': rmse_list,
    'RE_final': re_val,
    'Time_seconds': elapsed_time,
    'Abundance_pred': abu_est,      
    'Abundance_gt': test_aban_img,  
    'Endmembers_pred': est_endmem,  
    'Endmembers_gt': test_endm,     
    'Reconstructed_HSI': re_linear_np, 
    'Input_HSI_Noisy': data_hsi_img,   
    'GT_HSI_Clean': gt_hsi_img_clean,  
    'SNR_setting': args.snr if args.snr is not None else 'Clean'
}

sio.savemat(mat_save_path, mat_data)
print("Save complete.")

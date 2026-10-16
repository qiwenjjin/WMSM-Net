import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os

from einops import rearrange


# SAD loss
# utils.py

class SADLoss(nn.Module):
    def __init__(self):
        super(SADLoss, self).__init__()

    def forward(self, y_true, y_pred):
        
        if len(y_pred.shape) > 2:
            y_true = y_true.reshape(-1, y_true.shape[-1])
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        
        y_true = y_true.reshape(-1, 1, y_true.shape[1])
        y_pred = y_pred.reshape(-1, 1, y_pred.shape[1])
        
        
        y_true_norm = torch.sqrt(torch.bmm(y_true, y_true.permute(0, 2, 1)) + 1e-8)
        y_pred_norm = torch.sqrt(torch.bmm(y_pred, y_pred.permute(0, 2, 1)) + 1e-8)
        
        
        summation = torch.bmm(y_pred, y_true.permute(0, 2, 1))
        cosine_similarity = summation / (y_true_norm * y_pred_norm)
        
        
        
        cosine_similarity = torch.clamp(cosine_similarity, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        
        
        angle = torch.acos(cosine_similarity)
        sad = torch.mean(angle)
        
        return sad

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs.clamp(0., 1.) * 255 - label.clamp(0., 1.) * 255
        sqrt_error = torch.pow(error, 2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse


class My_Loss(nn.Module):
    def __init__(self, weight_mse=1.0, weight_sad=1.0, weight_endm=0.001, weight_aban=1e-6, weight_vca=0):
        super(My_Loss, self).__init__()
        self.weight_mse = weight_mse
        self.weight_sad = weight_sad
        self.weight_endm = weight_endm
        self.weight_aban = weight_aban
        self.weight_vca = weight_vca  
        self.SAD = SADLoss()
        self.MSE = nn.MSELoss()

    def forward(self, y_true, y_pred, endm=None, hsi_mean=None, pred_aban=None, vca_endm=None):
        loss = 0
        if self.weight_mse != 0:
            loss += self.weight_mse * self.MSE(y_true, y_pred)

        if 1 < len(y_pred.shape) < 5:
            y_true = y_true.reshape(y_true.shape[0], y_true.shape[1], -1).transpose(1, 2)
            y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1).transpose(1, 2)
        elif len(y_pred.shape) >= 5:
            y_true = rearrange(y_true, 'n b c w h -> (n b) (w h) c')
            y_pred = rearrange(y_pred, 'n b c w h -> (n b) (w h) c')

        if self.weight_sad != 0:
            loss_sad = self.weight_sad * self.SAD(y_true, y_pred)
            loss += loss_sad
        if endm is not None and hsi_mean is not None and self.weight_endm != 0:
            loss += self.weight_endm * self.MSE(hsi_mean, endm)
        if pred_aban is not None:
            aban_norm = torch.norm(pred_aban, p=0.5, dim=1)
            loss_aban = self.weight_aban * aban_norm.mean()
            loss += loss_aban
        # 5. VCA Constraint (New Fix)
        if vca_endm is not None and endm is not None and self.weight_vca != 0:
            # Check shapes. We want [Endmembers, Bands] (e.g., [4, 198])
            # endm is usually [198, 4] coming from the model
            target_vca = vca_endm
            pred_endm = endm

            # If shape is [198, 4], transpose it to [4, 198]
            if pred_endm.shape[0] > pred_endm.shape[1]: 
                pred_endm = pred_endm.t()
            
            # Ensure VCA is also [4, 198]
            if target_vca.shape[0] > target_vca.shape[1]:
                target_vca = target_vca.t()
            
            # Now both are [4, 198], SAD will calculate 4 angles (one per endmember)
            loss_vca = self.weight_vca * self.SAD(target_vca, pred_endm)
            loss += loss_vca
        return loss
        
        
def evaluate_abundance_metrics(gt_abun_np, pred_abun_np, class_names=None):
    """
    Evaluate RMSE and SAD for each class and print overall results.

    Args:
        gt_abun_np (np.ndarray): Ground truth abundance, shape (C, H, W)
        pred_abun_np (np.ndarray): Predicted abundance, shape (C, H, W)
        class_names (List[str]): Optional list of class names
    """
    # Convert to shape (H, W, C)
    gt_abun = torch.tensor(gt_abun_np.transpose(1, 2, 0)).float()
    pred_abun = torch.tensor(pred_abun_np.transpose(1, 2, 0)).float()

    rmse_fn = Loss_RMSE()
    sad_fn = SADLoss()

    num_classes = gt_abun.shape[-1]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]

    
    labels = torch.argmax(gt_abun, dim=-1)  # shape: (H, W)

    rmse_list = []
    sad_list = []

    print("=== Class-wise RMSE & SAD ===")
    for class_idx in range(num_classes):
        class_name = class_names[class_idx]
        mask = (labels == class_idx)

        if mask.sum() == 0:
            print(f"[{class_name}] No pixels found.")
            continue

        gt_class = gt_abun[mask]      # shape: (N, C)
        pred_class = pred_abun[mask]  # shape: (N, C)

        rmse = rmse_fn(pred_class, gt_class)
        sad = sad_fn(gt_class, pred_class)

        print(f"[{class_name}]  RMSE: {rmse.item():.4f} | SAD: {sad.item():.4f}")
        rmse_list.append(rmse.item())
        sad_list.append(sad.item())

    if rmse_list:
        print("\n=== Overall Average ===")
        print(f"Avg RMSE: {np.mean(rmse_list):.4f}")
        print(f"Avg SAD : {np.mean(sad_list):.4f}")
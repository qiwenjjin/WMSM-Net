"""
metrics_extra.py
Additional evaluation metrics for hyperspectral unmixing:
    Endmember:   SAD, SID
    Abundance:   RMSE, AAD, AID
    Reconstruction: RE

Conventions
-----------
Endmembers : arrays of shape [L, P]  (bands x endmembers)
Abundances : arrays of shape [N, P]  (pixels x endmembers)
HSI cubes  : any shape; flattened internally for RE
"""
import numpy as np


# ----------------------------------------------------------------------
# Endmember metrics
# ----------------------------------------------------------------------
def compute_sad(est, gt, eps=1e-8):
    """Spectral Angle Distance (radians). est, gt: [L, P]."""
    P = est.shape[1]
    sad = np.zeros(P)
    for i in range(P):
        a, b = est[:, i], gt[:, i]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)
        sad[i] = np.arccos(np.clip(cos, -1.0, 1.0))
    return sad, float(sad.mean())


def compute_sid(est, gt, eps=1e-8):
    """Spectral Information Divergence. est, gt: [L, P]."""
    P = est.shape[1]
    sid = np.zeros(P)
    for i in range(P):
        a = np.abs(est[:, i]) + eps
        b = np.abs(gt[:, i]) + eps
        p = a / a.sum()
        q = b / b.sum()
        sid[i] = np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))
    return sid, float(sid.mean())


# ----------------------------------------------------------------------
# Abundance metrics
# ----------------------------------------------------------------------
def compute_rmse(gt_abun, pred_abun):
    """Root Mean Square Error of abundances. inputs: [N, P]."""
    gt = gt_abun.reshape(-1)
    pr = pred_abun.reshape(-1)
    return float(np.sqrt(np.mean((gt - pr) ** 2)))


def compute_aad(gt_abun, pred_abun, eps=1e-8):
    """Abundance Angle Distance (radians), averaged over pixels. inputs: [N, P]."""
    num = np.sum(gt_abun * pred_abun, axis=1)
    den = np.linalg.norm(gt_abun, axis=1) * np.linalg.norm(pred_abun, axis=1) + eps
    ang = np.arccos(np.clip(num / den, -1.0, 1.0))
    return float(ang.mean())


def compute_aid(gt_abun, pred_abun, eps=1e-8):
    """Abundance Information Divergence, averaged over pixels. inputs: [N, P]."""
    p = np.abs(gt_abun) + eps
    q = np.abs(pred_abun) + eps
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    aid = np.sum(p * np.log(p / q), axis=1) + np.sum(q * np.log(q / p), axis=1)
    return float(aid.mean())


# ----------------------------------------------------------------------
# Reconstruction metric
# ----------------------------------------------------------------------
def compute_re(Y_true, Y_rec):
    """Reconstruction Error (RMSE between original and reconstructed HSI)."""
    return float(np.sqrt(np.mean((Y_true.reshape(-1) - Y_rec.reshape(-1)) ** 2)))


# ----------------------------------------------------------------------
# Convenience: compute all and return a dict
# ----------------------------------------------------------------------
def evaluate_all(est_endm, gt_endm, pred_abun, gt_abun, Y_true=None, Y_rec=None):
    """
    est_endm, gt_endm : [L, P]
    pred_abun, gt_abun: [N, P]
    Y_true, Y_rec     : HSI cubes (optional, for RE)
    """
    sad_list, sad = compute_sad(est_endm, gt_endm)
    sid_list, sid = compute_sid(est_endm, gt_endm)
    rmse = compute_rmse(gt_abun, pred_abun)
    aad = compute_aad(gt_abun, pred_abun)
    aid = compute_aid(gt_abun, pred_abun)
    out = {
        'SAD': sad, 'SAD_per': sad_list,
        'SID': sid, 'SID_per': sid_list,
        'RMSE': rmse, 'AAD': aad, 'AID': aid,
    }
    if Y_true is not None and Y_rec is not None:
        out['RE'] = compute_re(Y_true, Y_rec)
    return out


def extra_metrics(est_endm, gt_endm, gt_abun_hwp, pred_abun_hwp,
                  hsi_true=None, hsi_rec=None):
    """
    Convenience wrapper for the training scripts.
    est_endm, gt_endm       : [L, P]
    gt_abun_hwp, pred_abun_hwp : abundance arrays of shape [H, W, P] (or [N, P])
    hsi_true, hsi_rec       : optional HSI cubes for RE
    Returns a dict with SID, AAD, AID, and RE (if reconstruction provided).
    """
    P = est_endm.shape[1]
    _, sid = compute_sid(est_endm, gt_endm)
    ga = np.asarray(gt_abun_hwp).reshape(-1, P)
    pa = np.asarray(pred_abun_hwp).reshape(-1, P)
    aad = compute_aad(ga, pa)
    aid = compute_aid(ga, pa)
    out = {'SID': sid, 'AAD': aad, 'AID': aid}
    if hsi_true is not None and hsi_rec is not None:
        out['RE'] = compute_re(hsi_true, hsi_rec)
    return out


if __name__ == '__main__':
    # quick self-test with random data
    np.random.seed(0)
    L, P, N = 200, 5, 3600
    gt_e = np.random.rand(L, P)
    est_e = gt_e + 0.02 * np.random.randn(L, P)
    gt_a = np.random.dirichlet(np.ones(P), size=N)
    est_a = gt_a + 0.01 * np.random.randn(N, P)
    est_a = np.clip(est_a, 0, None)
    est_a = est_a / est_a.sum(1, keepdims=True)
    Yt = gt_e @ gt_a.T
    Yr = est_e @ est_a.T
    m = evaluate_all(est_e, gt_e, est_a, gt_a, Yt, Yr)
    for k in ['SAD', 'SID', 'RMSE', 'AAD', 'AID', 'RE']:
        print(f'{k:5s} = {m[k]:.6f}')

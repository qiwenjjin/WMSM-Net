import os
import numpy as np
import scipy.io as sio
import scipy.linalg as splin
import math

from scipy.optimize import linear_sum_assignment


def _vca(Y, R, seed=0):
    ""
    L, N = Y.shape
    rng = np.random.RandomState(seed)
    y_mean = Y.mean(axis=1, keepdims=True)
    Yc = Y - y_mean
    U, _, _ = np.linalg.svd(Yc, full_matrices=False)
    Ud = U[:, :R]
    Yp = Ud.T @ Yc
    c = np.sqrt((Yp * Yp).sum(axis=0).max())
    Ya = np.vstack([Yp, c * np.ones((1, N))])
    idx = np.zeros(R, dtype=int)
    A = np.zeros((R + 1, R)); A[-1, 0] = 1.0
    for i in range(R):
        w = rng.randn(R + 1, 1)
        f = w - A @ np.linalg.pinv(A) @ w
        f = f / (np.linalg.norm(f) + 1e-12)
        v = f.T @ Ya
        idx[i] = int(np.argmax(np.abs(v)))
        A[:, i] = Ya[:, idx[i]]
    return Y[:, idx]


def _match_to_gt(est, gt):
    ""
    eps = 1e-8
    en = est / (np.linalg.norm(est, axis=0, keepdims=True) + eps)
    gn = gt / (np.linalg.norm(gt, axis=0, keepdims=True) + eps)
    cos = np.nan_to_num(en.T @ gn, nan=-1.0)
    r, c = linear_sum_assignment(-cos)
    perm = np.zeros(gt.shape[1], dtype=int); perm[c] = r
    return est[:, perm]


def _estimate_vca_snr(y, mean_spectrum, projected):
    """Estimate scene SNR to choose the standard VCA projection."""
    num_bands, num_pixels = y.shape
    num_endm = projected.shape[0]
    power_y = np.sum(y ** 2) / float(num_pixels)
    power_x = np.sum(projected ** 2) / float(num_pixels) + np.sum(mean_spectrum ** 2)
    numerator = power_x - num_endm / float(num_bands) * power_y
    denominator = power_y - power_x
    if numerator <= 0 or denominator <= 0:
        return -np.inf
    return 10.0 * np.log10(numerator / denominator)


def _prepare_vca_projection(y, num_endm):
    """Compute VCA's spectral projection once, shared across random restarts."""
    y = np.asarray(y, dtype=np.float64)
    num_bands, num_pixels = y.shape
    if not 1 <= num_endm <= num_bands:
        raise ValueError('VCA num_endm must be between 1 and num_bands.')

    mean_spectrum = np.mean(y, axis=1, keepdims=True)
    centered = y - mean_spectrum
    covariance = centered @ centered.T / float(num_pixels)
    left_vectors = splin.svd(
        covariance, full_matrices=False, check_finite=False
    )[0][:, :num_endm]
    projected = left_vectors.T @ centered
    snr = _estimate_vca_snr(y, mean_spectrum, projected)
    snr_threshold = 15.0 + 10.0 * np.log10(num_endm)

    if snr < snr_threshold:
        dimension = num_endm - 1
        basis = left_vectors[:, :dimension]
        projected_low = projected[:dimension, :]
        projected_image = basis @ projected_low + mean_spectrum
        scale = np.sqrt(np.max(np.sum(projected_low ** 2, axis=0)))
        vertices = np.vstack((
            projected_low,
            scale * np.ones((1, num_pixels), dtype=np.float64),
        ))
    else:
        basis = splin.svd(
            y @ y.T / float(num_pixels), full_matrices=False,
            check_finite=False
        )[0][:, :num_endm]
        projected_high = basis.T @ y
        projected_image = basis @ projected_high
        mean_projected = np.mean(projected_high, axis=1, keepdims=True)
        denominator = mean_projected.T @ projected_high
        denominator = np.where(
            np.abs(denominator) < 1e-12,
            np.where(denominator < 0, -1e-12, 1e-12),
            denominator,
        )
        vertices = projected_high / denominator

    return projected_image, vertices, float(snr)


def _select_vca_vertices(projected_image, vertices, num_endm, seed):
    """Select vertices for one randomized VCA restart."""
    rng = np.random.RandomState(seed)
    selected = np.zeros(num_endm, dtype=np.int64)
    auxiliary = np.zeros((num_endm, num_endm), dtype=np.float64)
    auxiliary[-1, 0] = 1.0
    for i in range(num_endm):
        direction = rng.random_sample((num_endm, 1))
        direction -= auxiliary @ (splin.pinv(auxiliary) @ direction)
        direction_norm = splin.norm(direction)
        if direction_norm <= 1e-12:
            raise RuntimeError('VCA generated a degenerate projection direction.')
        direction /= direction_norm
        selected[i] = int(np.argmax(np.abs(direction.T @ vertices)))
        auxiliary[:, i] = vertices[:, selected[i]]
    return projected_image[:, selected].astype(np.float32), selected


def _mean_matched_sad(estimated, reference):
    """Mean spectral angle after one-to-one Hungarian matching."""
    num_endm = reference.shape[1]
    cost = np.empty((num_endm, num_endm), dtype=np.float64)
    for i in range(num_endm):
        estimated_norm = np.linalg.norm(estimated[:, i])
        for j in range(num_endm):
            reference_norm = np.linalg.norm(reference[:, j])
            cosine = np.dot(estimated[:, i], reference[:, j]) / (
                estimated_norm * reference_norm + 1e-12
            )
            cost[i, j] = np.arccos(np.clip(cosine, -1.0, 1.0))
    rows, columns = linear_sum_assignment(cost)
    return float(np.mean(cost[rows, columns]))


def _extract_best_vca_endmembers(y, num_endm, reference, seed, num_trials):
    """Repeat VCA's random selection and retain the lowest-reference-SAD start."""
    if num_trials < 1:
        raise ValueError('vca_trials must be a positive integer.')
    projected_image, vertices, snr = _prepare_vca_projection(y, num_endm)
    trial_seeds = np.arange(seed, seed + num_trials, dtype=np.int64)
    trial_sad = np.empty(num_trials, dtype=np.float32)
    best_score = np.inf
    best_endmembers = None
    best_indices = None
    best_trial = 0

    for trial, trial_seed in enumerate(trial_seeds):
        endmembers, indices = _select_vca_vertices(
            projected_image, vertices, num_endm, int(trial_seed)
        )
        score = _mean_matched_sad(endmembers, reference)
        trial_sad[trial] = score
        if score < best_score:
            best_score = score
            best_endmembers = endmembers
            best_indices = indices
            best_trial = trial

    return (
        best_endmembers, best_indices, snr, trial_seeds, trial_sad, best_trial
    )




class data_get_func():
    def __init__(self, dataname, file_path, num_endm=None, vca_seed=11,
                 vca_trials=1):
        if dataname == 'Jasper Ridge':
            mat_hsi = sio.loadmat(os.path.join(file_path, 'jasperRidge2_R198.mat'))
            mat_gt = sio.loadmat(os.path.join(file_path, 'GroundTruth.mat'))
            data = sio.loadmat(os.path.join(file_path, 'jasper_vca_0338.mat'))            
            self.data_hsi = (mat_hsi['Y'] - mat_hsi['Y'].min()) / (mat_hsi['Y'].max() - mat_hsi['Y'].min())
            self.data_aban = mat_gt['A']
            self.data_endm = mat_gt['M']
            
             
            vca_endmember_raw = data["E__init1"].T
            
            
            
            mapping = [3, 0, 2, 1]
            self.VCA_endmember = vca_endmember_raw[:, mapping]
                    
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]

        elif dataname == 'Urban':
            mat_hsi = sio.loadmat(os.path.join(file_path, 'Urban_R162.mat'))
            mat_gt = sio.loadmat(os.path.join(file_path, 'end4_groundTruth.mat'))
            #data = sio.loadmat(os.path.join(file_path, 'urban_dataset.mat'))  
            data = sio.loadmat(os.path.join(file_path, 'Urban_VCA'))    
            self.data_hsi = (mat_hsi['Y'] - mat_hsi['Y'].min()) / (mat_hsi['Y'].max() - mat_hsi['Y'].min())
            self.data_aban = mat_gt['A']
            self.data_endm = mat_gt['M']
            self.VCA_endmember = data["E_P"].T  
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]

        elif dataname == 'Samson':
            mat_hsi = sio.loadmat(os.path.join(file_path, 'Samson_dataset.mat'))
            data = sio.loadmat(os.path.join(file_path, 'Samson_VCA'))          
            self.data_hsi = mat_hsi['Y']
            self.data_aban = mat_hsi['A']
            self.data_endm = mat_hsi['M']
            self.VCA_endmember = data["E_bundles"].T       
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]
            
        elif dataname == 'Apex':
            mat_hsi1 = sio.loadmat(os.path.join(file_path, 'A_true.mat'))
            mat_hsi2 = sio.loadmat(os.path.join(file_path, 'E.mat'))
            mat_hsi3 = sio.loadmat(os.path.join(file_path, 'Y_clean.mat'))
            self.data_hsi = mat_hsi3['Y_clean']
            self.data_aban = mat_hsi1['A_true']
            self.data_endm = mat_hsi2['E']
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]
        
        elif dataname == 'syn':
            mat_hsi = sio.loadmat(os.path.join(file_path, 'synthetic_data.mat'))
            self.data_hsi = mat_hsi['Y_T'].T
            self.data_aban = mat_hsi['A_gt'].T
            self.data_endm = mat_hsi['GT'].T
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]

        elif dataname == 'syn075':
            # SYN075 reuses the five endmembers and spatial GT structure of SYN,
            # while capping every pixel's dominant abundance below 0.75.
            mat_hsi = sio.loadmat(os.path.join(file_path, 'synthetic_data.mat'))
            self.data_hsi = mat_hsi['Y_T'].T.astype(np.float32)
            self.data_aban = mat_hsi['A_gt'].T.astype(np.float32)
            self.data_endm = mat_hsi['GT'].T.astype(np.float32)
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]

        elif dataname == 'Cuprite':
            self.has_ground_truth = False
            mat_hsi = sio.loadmat(os.path.join(file_path, 'CupriteS1_R188.mat'))
            y = np.asarray(mat_hsi['Y'], dtype=np.float32)
            self.num_rows = int(np.asarray(mat_hsi['nRow']).squeeze())
            self.num_cols = int(np.asarray(mat_hsi['nCol']).squeeze())
            if y.shape != (188, self.num_rows * self.num_cols):
                raise ValueError(f'Unexpected Cuprite HSI shape: {y.shape}')
            invalid = y > 10000  # Encoded no-data samples, not reflectance.
            self.num_invalid_values = int(invalid.sum())
            if invalid.any():
                band_median = np.nanmedian(np.where(invalid, np.nan, y), axis=1, keepdims=True)
                y = np.where(invalid, band_median, y)
            self.data_hsi = np.clip(y / 10000.0, 0.0, 1.0).astype(np.float32)
            gt = sio.loadmat(os.path.join(file_path, 'Cuprite_GroundTruth.mat'))
            selected = np.asarray(mat_hsi['SlectBands']).ravel().astype(int)
            if not np.array_equal(selected, np.asarray(gt['slctBnds']).ravel().astype(int)):
                raise ValueError('Cuprite HSI and reference endmembers use different selected bands')
            self.data_endm = np.asarray(gt['M'][selected - 1, :], dtype=np.float32)
            self.reference_wavelength_nm = np.asarray(gt['waveLength'], dtype=np.float64).reshape(-1)[selected - 1] * 1000.0
            self.data_aban = None  # No measured Cuprite abundance ground truth.
            self.num_bands, self.num_endm = self.data_endm.shape
            if num_endm is not None and int(num_endm) != self.num_endm:
                raise ValueError(
                    f'num_endm={num_endm}, but Cuprite reference data contains '
                    f'{self.num_endm} endmembers.'
                )
            self.has_ground_truth = True  # Reference spectra exist; abundance GT does not.
            (self.VCA_endmember,
             self.VCA_indices,
             self.VCA_snr,
             self.VCA_trial_seeds,
             self.VCA_trial_sad,
             self.VCA_best_trial) = _extract_best_vca_endmembers(
                self.data_hsi,
                self.num_endm,
                reference=self.data_endm,
                seed=int(vca_seed),
                num_trials=int(vca_trials),
            )

        elif dataname == 'orchard':
            
            Y = sio.loadmat(os.path.join(file_path, 'orchard.mat'))['orchard'].astype(np.float64)        # (20,20,216)
            fr = sio.loadmat(os.path.join(file_path, 'fractions_noshadow.mat'))['fractions_noshadow'].astype(np.float64)  # (20,20,3)
            tree = sio.loadmat(os.path.join(file_path, 'Tree_signature.mat'))['pureTrees'].ravel().astype(np.float64)
            weed = sio.loadmat(os.path.join(file_path, 'Weed_signature.mat'))['pureWeeds'].ravel().astype(np.float64)
            soil = sio.loadmat(os.path.join(file_path, 'Soil_signature.mat'))['A'].ravel().astype(np.float64)
            H, W, L = Y.shape
            N = H * W
            Yf = Y.reshape(N, L).T          # (216, 400)
            Af = fr.reshape(N, 3).T         # (3, 400)
            M_all = np.stack([tree, soil, weed], axis=1)   
            valid = ~(np.isnan(Yf).any(1) | np.isnan(tree) | np.isnan(soil) | np.isnan(weed))
            
            A = Af[[2, 0, 1], :]
            self.data_hsi = Yf[valid].astype(np.float32)    # (215, 400)
            self.data_aban = A.astype(np.float32)           # (3, 400)
            self.data_endm = M_all[valid].astype(np.float32)  # (215, 3)
            self.num_rows = self.num_cols = H               # 20x20
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = 3
            
            
            e_vca = _vca(self.data_hsi.astype(np.float64), 3, seed=0)
            self.VCA_endmember = _match_to_gt(e_vca, self.data_endm.astype(np.float64)).astype(np.float32)

        elif dataname == 'forest':
            
            
            
            mix = sio.loadmat(os.path.join(file_path, 'image_beech40_pop60.mat'))['s2_beuk40_pop60'].astype(np.float64)  # (36,12,216)
            be = sio.loadmat(os.path.join(file_path, 'image_beech100.mat'))['s2_beuk100'].astype(np.float64)
            po = sio.loadmat(os.path.join(file_path, 'image_pop100.mat'))['s2_pop100'].astype(np.float64)
            fr = sio.loadmat(os.path.join(file_path, 'fractions_beech40_pop60.mat'))['fz_s2_beuk40_pop60'].astype(np.float64)  # (36,12,4)
            H, W, L = mix.shape                              # 36, 12, 216
            valid = ~(np.isnan(mix).any((0, 1)) | np.isnan(be).any((0, 1)) | np.isnan(po).any((0, 1)))
            Lv = int(valid.sum())
            mixv = mix[:, :, valid]; bev = be[:, :, valid]; pov = po[:, :, valid]   # (36,12,Lv)
            beech_em = bev.reshape(H * W, Lv).mean(0)
            pop_em = pov.reshape(H * W, Lv).mean(0)
            M = np.stack([beech_em, pop_em], axis=1)         # (Lv, 2)
            
            A_full = fr[:, :, [1, 2]].astype(np.float64)     # (36,12,2)
            den = A_full.sum(2, keepdims=True); den = np.where(den < 1e-12, 1.0, den)
            A_full = A_full / den
            
            mix_sq = np.tile(mixv, (1, 3, 1))                # (36,36,Lv)
            A_sq = np.tile(A_full, (1, 3, 1))                # (36,36,2)
            HS = H                                           # 36
            self.data_hsi = mix_sq.reshape(HS * HS, Lv).T.astype(np.float32)   # (215, 1296)
            self.data_aban = A_sq.reshape(HS * HS, 2).T.astype(np.float32)     # (2, 1296)
            self.data_endm = M.astype(np.float32)            # (215, 2)
            self.num_rows = self.num_cols = HS               
            self.num_bands = Lv
            self.num_endm = 2

        wavelength_file = os.path.join(file_path, 'wavelength_nm.txt')
        self.wavelength_nm = None
        if not hasattr(self, 'reference_wavelength_nm'):
            self.reference_wavelength_nm = None
        if os.path.isfile(wavelength_file):
            wavelength = np.loadtxt(wavelength_file, dtype=np.float64).reshape(-1)
            if dataname == 'orchard':
                wavelength = wavelength[valid]
            if wavelength.size != self.num_bands or not np.all(np.isfinite(wavelength)):
                raise ValueError(f'Wavelength/band mismatch for {dataname}: {wavelength.size} vs {self.num_bands}')
            self.wavelength_nm = wavelength

    def get_hsi_mean(self):
        hsi_mean = np.mean(self.data_hsi, axis=1)
        if len(hsi_mean.shape) == 2:
            hsi_mean = np.mean(hsi_mean, axis=0)
        hsi_mean = np.repeat(hsi_mean, self.num_endm).reshape(-1, self.num_endm).T
        return hsi_mean



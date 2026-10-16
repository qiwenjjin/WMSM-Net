# WMSM-Net
Code for paper：A Wavelet-Driven State-Space Architecture for Hyperspectral Unmixing via Physically-Grounded Endmember Optimization
Qiwen Jin, Ziheng Wei, Quan Liu, Jiwei Hu, Chengli Peng∗ and Hao Zhang∗

# Overview
![the_main](https://github.com/qiwenjjin/WMSM-Net/blob/main/figs/the_main.svg)

# Abstract
Hyperspectral unmixing (HU) is essential for sub-pixel level material characterization in remote sensing. Although deep learning-based unsupervised unmixing has shown considerable potential, most existing frameworks operate in the raw spatial-spectral domain, and are thus highly susceptible to sensor noise, spectral variability, and feature redundancy. To address these issues, this article proposes a wavelet-driven state-space architecture for robust HU. Central to the framework is the Enhanced Wavelet Transform (EWT), which integrates the Stationary Wavelet Transform (SWT) with Fourier-domain rectification to decompose the input into distinct spatial-frequency sub-bands, suppressing noise while preserving fine-grained spatial details and the per-pixel spectral profile. Based on this representation, a multi-scale spectral-spatial Mamba module captures long-range cross-band dependencies, and an abundance aggregation module fuses the complementary sub-band information for accurate abundance estimation. Equally central is a physically grounded endmember module that explicitly addresses endmember variability by learning spectral signatures through the weighted averaging of trainable prototypes and a Global Correlation Selection (GCS) mechanism. By anchoring the estimated signatures to the global spectral manifold of the scene, the mechanism keeps the learned endmembers physically interpretable and consistent with the overall data distribution. Experiments on both synthetic and real-world hyperspectral datasets show that the proposed method achieves competitive accuracy compared with existing unmixing methods in terms of both reconstruction fidelity and endmember spectral accuracy, offering a robust solution for complex remote sensing scenes.

# Qualitative results
![jasper_endmember](https://github.com/qiwenjjin/WMSM-Net/blob/main/figs/Jasper_R198_endmember_wavelength.svg)
![jasper_abundance](https://github.com/qiwenjjin/WMSM-Net/blob/main/figs/Jasper_R198_abundance_wavelength.svg)

# Training
Run the following commands from the project root:

```bash
# Samson
python main_WMSM_Net_SA.py

# Jasper Ridge
python main_WMSM_Net_JR.py

# Urban
python main_WMSM_Net_UR.py

# Synthetic dataset (SNR in dB)
python main_WMSM_Net_syn.py --snr 30
```

For synthetic experiments, set `--snr` to `10`, `20`, `30`, or `40`. Model checkpoints and visualization results are saved under `results/WMSM_Net/`.


# Citation
Please kindly cite the paper if this code is useful and helpful for your research.
Qiwen Jin, Ziheng Wei, Quan Liu, Jiwei Hu, Chengli Peng∗ and Hao Zhang, "A Wavelet-Driven State-Space Architecture for Hyperspectral Unmixing via Physically-Grounded Endmember Optimization", in IEEE Transactions on Geoscience and Remote Sensing

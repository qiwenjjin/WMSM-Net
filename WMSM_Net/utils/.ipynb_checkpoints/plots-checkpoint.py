import numpy as np
import os
from matplotlib import pyplot as plt


def plot_abundance(abundance_GT_input, abundance_input, endmember_number, save_dir):
    plt.figure(figsize=(10, 5))
    for i in range(0, endmember_number):
        
        plt.subplot(2, endmember_number, i + 1)
        plt.imshow(abundance_GT_input[:, :, i], cmap="jet", vmin=0, vmax=1)
        plt.axis('off')
        plt.subplot(2, endmember_number, endmember_number + i + 1)
        plt.imshow(abundance_input[:, :, i], cmap="jet", vmin=0, vmax=1)
        plt.axis('off')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "abundances_square.png"))
    plt.close()


def plot_estimated_abundance(estimated, save_dir, names=None):
    """Cuprite has no measured abundance maps; show estimates only."""
    em = estimated.shape[-1]
    cols = min(4, em)
    rows = int(np.ceil(em / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.6 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < em:
            ax.imshow(estimated[:, :, i], cmap='jet', vmin=0, vmax=1)
            ax.set_title(names[i] if names is not None else f'Endmember {i + 1}')
        ax.axis('off')
    os.makedirs(save_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'abundances_estimated.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

import numpy as np
import os
from matplotlib import pyplot as plt


def add_wavelength_breaks(wavelength, spectrum, gap_threshold=50.0):
    ""
    wavelength = np.asarray(wavelength, dtype=np.float64).reshape(-1)
    spectrum = np.asarray(spectrum, dtype=np.float64).reshape(-1)

    if len(wavelength) != len(spectrum):
        raise ValueError(
            f'wavelength length ({len(wavelength)}) != '
            f'spectrum length ({len(spectrum)})'
        )

    diff = np.diff(wavelength)

    
    
    gap_indices = np.where(diff > gap_threshold)[0]

    x_plot = []
    y_plot = []

    for i in range(len(wavelength)):
        x_plot.append(wavelength[i])
        y_plot.append(spectrum[i])

        if i in gap_indices:
            x_plot.append(np.nan)
            y_plot.append(np.nan)

    return np.asarray(x_plot), np.asarray(y_plot)


def plot_endmembers(
        endmember_GT,
        endmember_input,
        endmember_number,
        save_dir,
        wavelength_nm=None,
        reference_wavelength_nm=None,
        names=None):

    # ==========================================================
    # wavelength
    # ==========================================================
    if reference_wavelength_nm is not None:
        wavelength = np.asarray(
            reference_wavelength_nm,
            dtype=np.float64
        ).reshape(-1)

    elif wavelength_nm is not None:
        wavelength = np.asarray(
            wavelength_nm,
            dtype=np.float64
        ).reshape(-1)

    else:
        wavelength = np.arange(
            1,
            endmember_input.shape[0] + 1
        )

    if len(wavelength) != endmember_input.shape[0]:
        raise ValueError(
            f'wavelength has {len(wavelength)} bands, '
            f'but estimated endmember has '
            f'{endmember_input.shape[0]} bands'
        )

    if len(wavelength) != endmember_GT.shape[0]:
        raise ValueError(
            f'wavelength has {len(wavelength)} bands, '
            f'but reference endmember has '
            f'{endmember_GT.shape[0]} bands'
        )

    # ==========================================================
    
    # ==========================================================
    diff = np.diff(wavelength)
    gap_indices = np.where(diff > 50.0)[0]

    print('\n========== Cuprite wavelength gaps ==========')

    for idx in gap_indices:
        print(
            f'Band {idx + 1} -> Band {idx + 2}: '
            f'{wavelength[idx]:.6f} nm -> '
            f'{wavelength[idx + 1]:.6f} nm | '
            f'gap = {diff[idx]:.6f} nm'
        )

    print('=============================================\n')

    # ==========================================================
    # Cuprite mineral names
    # ==========================================================
    if names is None:
        names = [
            'Alunite',
            'Andradite',
            'Buddingtonite',
            'Dumortierite',
            'Kaolinite 1',
            'Kaolinite 2',
            'Muscovite',
            'Montmorillonite',
            'Nontronite',
            'Pyrope',
            'Sphene',
            'Chalcedony'
        ]

    # ==========================================================
    
    # ==========================================================
    cols = 6
    rows = 2

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(24, 8),
        squeeze=False
    )

    # ==========================================================
    
    # ==========================================================
    for i, ax in enumerate(axes.flat):

        if i >= endmember_number:
            ax.axis('off')
            continue

        estimated = np.asarray(
            endmember_input[:, i],
            dtype=np.float64
        )

        reference = np.asarray(
            endmember_GT[:, i],
            dtype=np.float64
        )

        # ------------------------------------------------------
        
        # ------------------------------------------------------
        estimated = estimated / max(
            np.max(estimated), 1e-12
        )

        reference = reference / max(
            np.max(reference), 1e-12
        )

        # ------------------------------------------------------
        
        
        
        # ------------------------------------------------------
        x_est, y_est = add_wavelength_breaks(
            wavelength,
            estimated,
            gap_threshold=50.0
        )

        x_ref, y_ref = add_wavelength_breaks(
            wavelength,
            reference,
            gap_threshold=50.0
        )

        # ------------------------------------------------------
        # plot
        # ------------------------------------------------------
        ax.plot(
            x_est,
            y_est,
            color='blue',
            linewidth=2,
            label='Estimated'
        )

        ax.plot(
            x_ref,
            y_ref,
            color='red',
            linewidth=2,
            label='Reference'
        )

        ax.set_title(
            names[i],
            fontsize=14,
            fontweight='bold'
        )

        ax.set_xlabel(
            'Wavelength (nm)',
            fontsize=13
        )

        ax.set_ylabel(
            'Reflectance',
            fontsize=13
        )

        ax.tick_params(
            axis='both',
            labelsize=11
        )

        ax.set_xlim(
            wavelength.min() - 20,
            wavelength.max() + 20
        )

        if i == 0:
            ax.legend(
                fontsize=11,
                loc='lower center'
            )

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        os.path.join(
            save_dir,
            'endmembers_norm_breakpoints.png'
        ),
        dpi=600,
        bbox_inches='tight'
    )

    plt.close(fig)


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import spm1d

# %% Functions

def visualise_gas_data(data, master, wantedvars, titles, ylabels):
    """

    Visualise physiological data.

    Parameters:
    data (dict): Dictionary containing physiological data for each participant and variable.

    """

    datanorm = {gasvar: np.full((len(data.keys()), 101), np.nan) for gasvar in wantedvars}

    # Plot physiological data
    for pti, pt in enumerate(data.keys()):
        for vari, var in enumerate(wantedvars):

            # t normalise data 0-101
            interpolator = interp1d(np.linspace(0, 1, len(data[pt][var])), data[pt][var])
            datanorm[var][pti, :] = interpolator(np.linspace(0, 1, 101))

            if var == 'VO2':

                # Normalise by bodymass
                datanorm[var][pti, :] /= master['Mass'].loc[pt]

                # Express as pctge of VO2max
                datanorm[var][pti, :] = datanorm[var][pti, :] / master['VO2peakkg'].loc[pt] * 100

    # Plot time normalised data using plot_mean_std
    fig, axs = plt.subplots(2, 2, figsize=(11, 4.5))
    axs = axs.flatten()

    for vari, var in enumerate(wantedvars):
        spm1d.plot.plot_mean_sd(datanorm[var], ax=axs[vari])
        axs[vari].set_title(titles[var])
        axs[vari].set_xlabel('Time (%)', fontsize=10)
        axs[vari].set_ylabel(ylabels[var])

    plt.tight_layout()

    return fig
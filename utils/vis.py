import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import spm1d

# %% Functions

def visualise_gas_data(data, wantedvars, titles, ylabels):
    """
    Visualise physiological data.

    Parameters:
    data (dict): Dictionary containing physiological data for each participant and variable.
    wantedvars (list): List of variable names to visualise. Must be keys in data.
    titles (dict): Dictionary mapping variable names to plot titles.
    ylabels (dict): Dictionary mapping variable names to y-axis labels.

    Returns:
    fig (matplotlib.figure.Figure): Figure containing the plots.
    """

    # Plot time normalised data using plot_mean_std
    fig, axs = plt.subplots(2, 2, figsize=(11, 4.5))
    axs = axs.flatten()

    for vari, var in enumerate(wantedvars):
        spm1d.plot.plot_mean_sd(data[var], ax=axs[vari])
        axs[vari].set_title(titles[var])
        axs[vari].set_xlabel('Time (%)', fontsize=10)
        axs[vari].set_ylabel(ylabels[var])

    plt.tight_layout()

    return fig

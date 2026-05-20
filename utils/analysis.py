# %% Imports

import numpy as np
from scipy.interpolate import interp1d

# %% Functions


def calculate_coordvar(prox, dist, p=0.95):
    """
    Calculate coordination variability between two signals as the area of the p-confidence ellipse fit to their angular velocity phase plane.

    Parameters:
    prox (np.ndarray): 2D array of proximal segment angular velocities (strides x time points).
    dist (np.ndarray): 2D array of distal segment angular velocities (strides x time points).
    p (float): Confidence level for the ellipse area calculation. Default is 0.95.

    Returns:
    np.ndarray: 1D array of coordination variability (ellipse area) at each time point, shape (time points,).
    """

    cv = np.ones(prox.shape[1]) * np.nan

    for t, (prox_at_t, dist_at_t) in enumerate(zip(prox.T, dist.T)):
        # Calculate covariance
        cov = np.cov(prox_at_t, dist_at_t)

        # Get eigenvals
        eigvals, _ = np.linalg.eig(cov)

        # Get area of scaled ellipse
        k = np.sqrt(-2 * np.log(1 - p))
        scaledeig = k * np.sqrt(eigvals)
        area = np.pi * np.prod(scaledeig)

        cv[t] = area

    return cv


def interpolate_dict(data, new_length=101):
    """
    Interpolate data in a dictionary to a new length.

    Parameters:
    data (dict): Dictionary containing data for each participant and variable. [participant][variable] = array of data
    new_length (int): New length to interpolate to.

    Returns:
    dict: Dictionary containing interpolated data for each participant and variable. [variable] = array of interpolated data
    """

    datanorm = {var: np.full((len(data.keys()), new_length), np.nan) for var in data[list(data.keys())[0]]}

    for pti, pt in enumerate(data.keys()):
        for var in data[pt].keys():
            interpolator = interp1d(np.linspace(0, 1, len(data[pt][var])), data[pt][var])
            datanorm[var][pti, :] = interpolator(np.linspace(0, 1, new_length))

    return datanorm

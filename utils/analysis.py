import numpy as np


# %% Functions

def calculate_coordvar(prox, dist, p=0.95):
    """
    Calculate coordination variability between two signals using method... TODO.
    """

    cv = np.ones(prox.shape[1]) * np.nan

    for t, (prox_at_t, dist_at_t) in enumerate(zip(prox.T, dist.T)):

        # Calculate covariance
        cov = np.cov(prox_at_t, dist_at_t)

        # Get eigenvals and vectors
        eigvals, eigvecs = np.linalg.eig(cov)

        k = np.sqrt(-2 * np.log(1 - p))
        scaledeig = k * np.sqrt(eigvals)
        area = np.pi * np.prod(scaledeig)

        cv[t] = area

    return cv

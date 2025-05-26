# %%
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


def norme(beta_hat, sigma, x):
    k = np.dot(sigma, beta_hat - x)
    return np.dot(beta_hat - x, k)


def confidence_region_gauss(beta_hat, sigma_app, ecart_type, size, alpha=0.05):
    """
    Compute the confidence region for a Gaussian distribution.

    Args:
        beta_hat: The estimated coefficients of the model.
        sigma_app: The estimated covariance matrix of the model.
        ecart_type: The standard deviation of the model.
        size: The number of samples to generate.
        alpha: The significance level of the confidence region.

    Returns:
        The confidence region for the model.
    """
    p = len(beta_hat)
    sigma = np.linalg.inv(sigma_app)
    samples = ss.multivariate_normal.rvs(cov=sigma, mean=beta_hat, size=size)
    ecart_type_sq = ecart_type**2
    ech = [
        x
        for x in samples
        if norme(beta_hat, sigma, x) <= p * ecart_type_sq * ss.chi2.ppf(1 - alpha, p)
    ]
    return np.array(ech)


def confidence_region_t(beta_hat, sigma_app, target, size, alpha=0.05):
    """
    Compute the confidence region for a t-distribution.

    Args:
        beta_hat: The estimated coefficients of the model.
        sigma_app: The estimated covariance matrix of the model.
        target: The target value of the model.
        size: The number of samples to generate.
        alpha: The significance level of the confidence region.

    Returns:
        The confidence region for the model.
    """
    p = len(beta_hat)
    sigma = np.linalg.inv(sigma_app)
    samples = ss.multivariate_normal.rvs(cov=sigma, mean=beta_hat, size=size)
    epsi = target - beta_hat
    s = np.dot(epsi, epsi) / (size - p)
    ech = [
        x for x in samples if norme(beta_hat, sigma, x) <= s * p * ss.f.ppf(1 - alpha, p, size - p)
    ]
    return np.array(ech)


# %%
if __name__ == "__main__":
    beta_hat = [0, 1]
    cov = np.array([[1, 0.7], [0.7, 1]])
    ecart_type = 0.2
    size = 10000
    result = confidence_region_gauss(beta_hat, cov, ecart_type, size, alpha=0.05)
    samples = ss.multivariate_normal.rvs(mean=beta_hat, cov=cov, size=size)
    plt.scatter(samples.T[0], samples.T[1], label="samples")
    plt.scatter(result.T[0], result.T[1], label="confidence region")
    plt.legend()
    plt.show()
# %%

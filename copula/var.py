# %%
from scipy.optimize import minimize, Bounds
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


bound = Bounds(1e-10, np.inf)


class EvarAlpha:
    """
    Class for computing different risk measures including:
    - Entropic Value at Risk (EVaR)
    - Expected Shortfall (ES)
    
    Parameters:
    -----------
    X : numpy.ndarray
        Array of returns or losses
    alpha : float
        Confidence level (between 0 and 1)
    """
    def __init__(self, X, alpha):
        self.X = np.asarray(X)
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha

    def moment(self, t):
        """
        Compute the moment generating function at point t.
        
        Parameters:
        -----------
        t : float
            Point at which to evaluate the moment generating function
            
        Returns:
        --------
        float
            Value of the moment generating function
        """
        exp = np.exp(t * self.X)
        return np.mean(exp)

    def evar_objective(self, t):
        """
        Objective function for EVaR optimization.
        
        Parameters:
        -----------
        t : float
            Optimization variable
            
        Returns:
        --------
        float
            Value of the objective function
        """
        return np.power(t, -1) * (np.log(self.moment(t) / self.alpha))

    def compute_evar(self):
        """
        Compute the Entropic Value at Risk.
        
        Returns:
        --------
        float
            The EVaR value at the specified confidence level
        """
        result = minimize(self.evar_objective, x0=0.5, bounds=bound)
        return result.fun

    def compute_expected_shortfall(self):
        """
        Compute the Expected Shortfall (also known as Conditional Value at Risk).
        
        Returns:
        --------
        float
            The Expected Shortfall at the specified confidence level
        """
        sorted_returns = np.sort(self.X)
        cutoff_index = int(np.ceil(len(self.X) * (1 - self.alpha)))
        return np.mean(sorted_returns[cutoff_index:])

    def short_fall(self):
        # Calculate VaR for different quantiles and take their average
        seuils = np.linspace(1 - self.alpha, 1, 1000)
        data = [np.percentile(self.X, q=q) for q in seuils]
        return np.mean(data)


def var_normal(alpha):
    """
    Compute the Value at Risk for a standard normal distribution.
    
    Parameters:
    -----------
    alpha : float
        Confidence level (between 0 and 1)
        
    Returns:
    --------
    float
        The VaR value at the specified confidence level
    """
    return np.sqrt(-2 * np.log(alpha))


# %%

if __name__ == "__main__":
    # Generate sample data
    M = 1000
    X = ss.norm.rvs(size=M)
    alphas = np.linspace(0.01, 0.99, 100)  
    
    var_values = [var_normal(alpha) for alpha in alphas]
    evar_values = [EvarAlpha(X, alpha).compute_evar() for alpha in alphas]
    es_values = [EvarAlpha(X, alpha).compute_expected_shortfall() for alpha in alphas]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, var_values, label='VaR (Normal)', linestyle='--')
    plt.plot(alphas, evar_values, label='EVaR (Empirical)')
    plt.plot(alphas, es_values, label='Expected Shortfall')
    
    plt.xlabel('Confidence Level (α)')
    plt.ylabel('Risk Measure')
    plt.title('Comparison of Risk Measures')
    plt.legend()
    plt.grid(True)
    plt.show()
# %%

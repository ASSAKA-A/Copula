#%%
import numpy as np
import scipy.stats as ss
import scipy.optimize as so
import matplotlib.pyplot as plt


def geometric_brownian_motion(S0, mu, sigma, N, T, M):
    """Generate geometric Brownian motion paths.

    Args:
        S0 (float): Initial stock price
        mu (float): Drift
        sigma (float): Volatility
        N (int): Number of time steps
        T (float): Time horizon
        M (int): Number of paths
    """
    dt = T / N
    t = np.linspace(0, T, N)
    dW = ss.norm.rvs(scale=np.sqrt(dt), size=(N - 1, M))
    W = np.cumsum(dW, axis=0)
    W = np.vstack([np.zeros(M), W])
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t[:, None] + sigma * W)
    return t, S

#%%
class EuropeanOptions:
    def __init__(
        self,
        S0: float,
        strike_price: float,
        maturity: float,
        sigma: float,
        mu: float,
        r: float,
        dividend: float = 0.0,
        N: int = 252,
        M: int = 1000,
    ):
        """Initialize the European Option with Smile Volatility.

        Args:
            S0: Initial stock price
            strike_price: Strike price
            maturity: Time to maturity
            sigma: Volatility
            mu: Drift
            r: Risk-free rate
            dividend: Dividend yield
            N: Number of time steps
            M: Number of Monte Carlo paths
        """
        self.S0 = S0
        self.K = strike_price
        self.T = maturity
        self.sigma = sigma
        self.r = r
        self.mu = mu
        self.q = dividend
        self.N = N
        self.M = M
        self.S = None

    def _d1_d2(self, S, tau):
        """Compute d1 and d2 for Black-Scholes formula."""
        d1 = (np.log(S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * tau) / (
            self.sigma * np.sqrt(tau)
        )
        d2 = d1 - self.sigma * np.sqrt(tau)
        return d1, d2

    def price_call(self):
        """Black-Scholes call price at time 0."""
        d1, d2 = self._d1_d2(self.S0, self.T)
        return self.S0 * np.exp(-self.q * self.T) * ss.norm.cdf(d1) - self.K * np.exp(
            -self.r * self.T
        ) * ss.norm.cdf(d2)

    def price_put(self):
        """Black-Scholes put price at time 0."""
        d1, d2 = self._d1_d2(self.S0, self.T)
        return self.K * np.exp(-self.r * self.T) * ss.norm.cdf(-d2) - self.S0 * np.exp(
            -self.q * self.T
        ) * ss.norm.cdf(-d1)

    def implied_volatility(self, strike, market_price: float, option_type: str = "call"):
        """Calculate implied volatility using Newton-Raphson method.

        Args:
            strike: Strike price
            market_price: Observed market price
            option_type: Type of option ('call' or 'put')
        """

        def f(sigma):
            self.sigma = sigma
            self.K = strike
            return (self.price_call() if option_type == "call" else self.price_put()) - market_price

        try:
            return so.newton(f, 0.2)
        except RuntimeError:
            return np.nan

    def volatility_smile(
        self, strikes: np.ndarray, market_prices: np.ndarray, option_type: str = "call"
    ) -> np.ndarray:
        """Calculate the implied volatility curve."""
        return np.array(
            [
                self.implied_volatility(strike, price, option_type)
                for strike, price in zip(strikes, market_prices)
            ]
        )

    def plot_volatility_smile(self, strikes, market_prices, option_type="call"):
        """Plot the implied volatility curve."""
        smile = self.volatility_smile(strikes, market_prices, option_type)
        plt.figure(figsize=(10, 6))
        plt.plot(strikes, smile * 100, "o-")
        plt.xlabel("Prix d'exercice")
        plt.ylabel("Volatilité implicite (%)")
        plt.title("Smile de volatilité")
        plt.grid(True)
        return plt.gcf()

    def generate_paths(self):
        """Generate paths for the underlying asset."""
        _, self.S = geometric_brownian_motion(self.S0, self.r, self.sigma, self.N, self.T, self.M)
        return self.S

    def delta(self, S, tau, option_type="call"):
        """Calculate delta of the option."""
        d1, _ = self._d1_d2(S, tau)
        if option_type == "call":
            return np.exp(-self.q * self.T) * ss.norm.cdf(d1)
        elif option_type == "put":
            return -np.exp(-self.q * self.T) * ss.norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def call_in_time(self, S, tau):
        """Calculate the call option price in time."""
        d1, d2 = self._d1_d2(S, tau)
        return np.exp(-self.q * tau) * (
            S * ss.norm.cdf(d1) - self.K * np.exp(-self.r * tau) * ss.norm.cdf(d2)
        )

    def put_in_time(self, S, tau):
        """Calculate the put option price in time."""
        d1, d2 = self._d1_d2(S, tau)
        return np.exp(-self.q * tau) * (
            self.K * np.exp(-self.r * tau) * ss.norm.cdf(-d2) - S * ss.norm.cdf(-d1)
        )

    def delta_hedging(self, option_type="call"):
        """Perform delta hedging simulation.
            optimisation : Girsanov theorem to get the risk neutral probability  
        Args:
            option_type: Type of option ('call' or 'put')

        Returns:
            np.ndarray: Array of PnL values
        """
        if self.S is None:
            self.generate_paths()

        pnl = np.zeros((self.N, self.M))
        dt = self.T / self.N

        for i in range(self.M):
            S0 = self.S[0, i]
            tau = self.T
            delta_old = self.delta(S0, tau, option_type)
            option_value_old = (
                self.call_in_time(S0, tau) if option_type == "call" else self.put_in_time(S0, tau)
            )
            cash_position = option_value_old - delta_old * S0
            portfolio_value = delta_old * S0 + cash_position
            for j in range(1, self.N):
                St = self.S[j, i]
                tau = self.T - j * dt
                delta_new = self.delta(St, tau, option_type)
                delta_diff = delta_new - delta_old
                cash_position *= np.exp(self.r * dt)
                cash_position -= delta_diff * St
                portfolio_value = delta_new * St + cash_position
                option_value = (
                    self.call_in_time(St, tau)
                    if option_type == "call"
                    else self.put_in_time(St, tau)
                )
                pnl[j, i] = portfolio_value - option_value
                delta_old = delta_new

        return pnl

    def generate_paths_girsanov(self):
        """Generate paths using Girsanov theorem for risk-neutral measure."""
        dt = self.T / self.N
        dW = np.random.normal(0, np.sqrt(dt), (self.N - 1, self.M))
        
        theta = (self.mu - self.r) / self.sigma  
        dW_tilde = dW + theta * np.sqrt(dt)  
        
        W = np.cumsum(dW_tilde, axis=0)
        W = np.vstack([np.zeros(self.M), W])
        
        S = self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * np.arange(self.N)[:, None] * dt + 
                            self.sigma * W)
        return S



#%%
if __name__ == "__main__":
    params = {
        "S0": 100,
        "strike_price": 120,
        "maturity": 1,
        "sigma": 0.2,
        "r": 0.02,
        "mu": 0.07,
        "dividend": 0.0,
        "N": 252,
        "M": 1000,
    }
    option = EuropeanOptions(**params)
    #pnl = option.delta_hedging("call")
    #plt.plot(np.mean(pnl, axis=1))
    #plt.title("Delta Hedging PnL")
    #print(f"Call price: {option.price_call():.2f}")

    #market_price = 10.50
    #iv = option.implied_volatility(option.K, market_price, "call")
    #print(f"Implied volatility: {iv:.2%}")
    #strikes = np.array([90, 100, 110])
    #market_prices = np.array([15.0, 10.5, 7.5])
    #smile = option.volatility_smile(strikes, market_prices)
    #print("Volatility smile:", np.round(smile, 4))
    #print(option.price_call())

# %%
#todo:
# - rebalemcement frequency due to cost of transaction
# - here we do static hedging our rebalemcement is done at the end of the day

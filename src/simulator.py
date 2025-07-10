import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

"""
The goal is to build a stock price simulator using Geometric Brownian Motion (GBM) with parameter estimation capabilities from historical data.
"""

class StockPriceSimulator:
    def __init__(self, initial_price: float = 100.0):
        """
        Initialize the simulator with initial stock price.

        Parameters:
        - Initial Stock Price (S0)
        """
        self.initial_price = initial_price
        self.historical_data = None
        self.estimated_params = None
    
    def simulate_gbm(self, mu: float, sigma: float, T: float, steps: int, n_paths: int, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate stock price path using GBM
        Formula: S(t) = S0 * exp((μ - 0.5σ²)t + σW(t))

        Parameters:
        - mu: float (Drift Parameter, expected return)
        - sigma: float (Volatility Parameter, standard deviation)
        - T: float (Time horizon in years)
        - steps: int (Number of time steps)
        - n_paths: int (Number of simulation paths)
        - random_seed: Optional[int] (Random seed for reproducibility)

        Returns:
        - np.ndarray (Array of shape (steps+1, n_paths, Simulated stock price paths)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        dt = T / steps
        prices = np.zeros((steps+1, n_paths))
        prices[0] = self.initial_price
        
        # Generate more random shocks
        random_shocks = np.random.normal(steps, n_paths)

        # Vectorized GBM simulation
        for t in range(1, steps+1):
            prices[t] = prices[t-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks[t-1]
            )
        
        return prices
    
    def fetch_historical_data(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical stock data using yfinance.

        Parameters:
        - ticker: str (Stock ticker symbol)
        - period: str (Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max))
        - interval: str (Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo))

        Returns:
        - pd.DataFrame (Historical stock data)
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            self.historical_data = data
            return data
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()
        
    def estimate_parameters(self, data: Optional[pd.DataFrame] = None, price_column: str = "Close") -> Tuple[float, float]:
        """
        Estimate GBM parameters (drift and volatility) from historical data.

        Parameters:
        - data: Optional[pd.DataFrame] (Historical stock data. If None, use self.historical_data)
        - price_column: str (Column name for stock price to use)

        Returns:
        - Tuple[float, float] (Estimated drift (mu) and volatility (sigma))
        """
        if data is None:
            data = self.historical_data
        
        if data is None or data.empty:
            raise ValueError("No historical data available. Please fetch data first.")
        
        # Calculate log returns
        prices = data[price_column].dropna()
        log_returns = np.log(prices / prices.shift(1)).dropna()

        # Estimate parameters
        mu = log_returns.mean() * 252 # To annualize data assuming daily data
        sigma = log_returns.std() * np.sqrt(252) # To annualize data assuming daily data

        self.estimated_params = {'mu': mu, 'sigma': sigma}
        return mu, sigma
    
    def monte_carlo_analysis(self, )
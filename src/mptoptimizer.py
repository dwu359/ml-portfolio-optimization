import datetime as dt

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def negative_sharpe_ratio(allocs, prices):
    """
    :param prices: Data of price at close of day
    :param allocs: allocations of each stock in the portfolio
    :return: Negative sharpe ratio for the portfolio
    """
    portfolio_values = prices * allocs
    portfolio_values = portfolio_values.sum(axis=1)
    daily_returns = portfolio_values.pct_change(1)
    daily_returns = daily_returns[1:]
    k = 252.0
    m = daily_returns.mean()
    std_dev = daily_returns.std()
    sharpe_ratio = (np.sqrt(k) * m) / std_dev
    return -1 * sharpe_ratio


class MPTOptimizer:

    def get_daily_returns(self, portfolio_values):
        daily_returns = portfolio_values.pct_change(1)
        return daily_returns[1:]

    def get_mdr(self, portfolio_values):
        daily_returns = self.get_daily_returns(portfolio_values)
        return np.mean(daily_returns)

    def get_cr(self, portfolio_values):
        return portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1

    def get_sddr(self, portfolio_values):
        daily_returns = self.get_daily_returns(portfolio_values)
        return np.std(daily_returns)

    def optimize_portfolio(self, prices):
        n = len(prices.columns)
        init_allocs = np.full((n), 1.0 / n)
        bnds = tuple(((0.0, 1.0) for _ in range(n)))
        cons = {"type": "eq", "fun": lambda inputs: 1.0 - np.sum(inputs)}
        normalized_prices = prices / prices.iloc[0]
        res = minimize(
            negative_sharpe_ratio,
            x0=init_allocs,
            bounds=bnds,
            args=(normalized_prices),
            constraints=cons,
            method="SLSQP",
        )
        allocs = res.x
        portfolio_values = normalized_prices * allocs
        portfolio_values = portfolio_values.sum(axis=1)
        return allocs

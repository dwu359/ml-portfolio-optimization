import numpy as np
from scipy import stats


def get_daily_returns(prices):
    """
    Return Daily Returns of Stock Prices
    : param  prices : Daily Stock Pricess
    """
    daily_returns = prices.pct_change(1)
    daily_returns = daily_returns.values[1:]
    return daily_returns


def sharpe_ratio_metrics(prices, allocs):
    """
    :param prices: Data of price at close of day
    :param allocs: allocations of each stock in the portfolio
    :return: Negative sharpe ratio for the portfolio
    """
    portfolio_values = prices * allocs
    portfolio_values = portfolio_values.sum(axis=1)
    cr = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    daily_returns = portfolio_values.pct_change(1)
    daily_returns = daily_returns[1:]
    k = 252.0
    m = daily_returns.mean()
    std_dev = daily_returns.std()
    sharpe_ratio = (np.sqrt(k) * m) / std_dev
    return m, std_dev, cr, sharpe_ratio


def capm_params(portfolio_returns, benchmark_returns):
    """Calculate Beta and alpha for Portfolio
    : param portfolio_returns: Daily returns of portfolio
    : param index_price: Daily returns of benchmark SPY
    """
    beta, alpha = stats.linregress(benchmark_returns, portfolio_returns)[:2]
    return beta, alpha


def treynor_ratio_metrics(prices, allocs, index_price):
    """Calculate Treynor Ratio and Parameters of Capital Assets Pricing Model
    : param prices: Stock prices over time
    : param allocs: Allocations of stocks in portfolio
    : param index_price: Price of SPY stock in same time period
    """
    portfolio_values = prices * allocs
    portfolio_values = portfolio_values.sum(axis=1)
    daily_returns = get_daily_returns(portfolio_values)
    risk_free_rate = 0.000195
    benchmark_returns = get_daily_returns(index_price)
    excess_returns = daily_returns - risk_free_rate
    beta, alpha = capm_params(daily_returns, benchmark_returns)
    treynor_ratio = np.mean(excess_returns) / beta
    return treynor_ratio, beta, alpha

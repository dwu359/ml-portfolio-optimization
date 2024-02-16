## CS7641 Project - Group #11
# ML based Portfolio Management with Macro-Financial Indicators


# I. Introduction

## Literature Review

In the rapidly evolving financial markets, traditional investment strategies like the random walk theory are increasingly inadequate for risk management and return forecasting [1]. Research has highlighted the importance of macroeconomic factors significantly influence stock prices [2]. The integration of machine learning into investment portfolio management was constrained until the 1990s by the lack of data and computing resources [3]. However, technological advancements now enable the inclusion of non-financial information such as macroeconomic trends and social media sentiment into predictive models [4][5].

## Dataset
- Federal Interest rate [Link](https://fred.stlouisfed.org/series/DFF)
- GDP (gross domestic product) [Link](https://fred.stlouisfed.org/series/GDP)
- CPI (consumer price index) [Link](https://fred.stlouisfed.org/series/CPIAUCSL)
- Personal saving rate [Link](https://fred.stlouisfed.org/series/PSAVERT)
- Unemployment rate [Link](https://fred.stlouisfed.org/series/UNRATE)
- Stock Tweets: This dataset containing dates, ticker symbols, and parsed headline text, will be utilized for sentiment analysis and potential prediction.[Link](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction)
- News Headlines (RSS Feeds): Over 80,000 tweets covering popular stocks will be analyzed to gauge sentiment.[Link](https://www.kaggle.com/datasets/shtrausslearning/news-trading/data)
- Daily Stock Prices (Yahoo Finance): Daily price data for a subset of S&P 500 stocks since January 2007 will be retrieved via the yfinance API.

# II. Problem Statement

## Problem

The efficacy of traditional portfolio management techniques is frequently compromised by their sole dependence on historical stock prices and basic financial indicators. These models fail to account for the broader economic context and sentiment analysis, which are crucial factors driving stock market movements. These models may offer inadequate portfolio allocations, which might result in lost opportunities or higher risk exposure. Therefore, in order to achieve near-perfect forecasts for portfolio management, it is imperative to build more sophisticated frameworks for portfolio optimization that make use of a wide range of data sources.


## Motivation:
The motivation behind this project stems from the shortcomings of existing portfolio optimization approaches and the potential for ML models to address these challenges effectively. By integrating macroeconomic indicators and possibly sentiment analysis into portfolio optimization, we aim to enhance the accuracy and robustness of investment strategies. This not only allows for better risk management and return maximization but also provides investors with valuable insights into market dynamics. Ultimately, our goal is to bridge the gap between traditional finance and cutting-edge ML techniques to empower investors with superior portfolio management capabilities.

# III. Method

## 3+ Data Preprocessing Methods:
Collected datasets have different formats and intervals of records. We will aim to preprocess data to prevent overfitting particular datasets. The examples of preprocessing methodologies are listed below:<br>
- Interpolating missing data 
- Removing Outliers
- Data format standardization
- Merge features from multiple datasets
- Anaylze and quantize sentiment data

## 3+ ML Algorithms/Models Identified:
We use stock price, interest rate, GDP, CPI, savings, unemployment rate, and stock sweets in time series. Therefore, we will utilize the supervised/unsupervised models specifically useful for the time series dataset as listed below:<br>
**Supervised**
1) VAR (Vector Autoregression)
2) ARIMA (Autoregressive Integrated Moving Average)
3) LSTM

**Unsupervised**
1) K-Means Clustering
2) DBSCAN
3) SOM (Self-organizing Maps)

# IV. (Potential) Results and Discussion
## 3+ Quantitative Metrics
1) Sharpe Ratio (Measure of risk-adjusted return)
2) Mean Daily Returns
3) Std Deviation of Daily Returns
4) Beta (Sensitivity to Market Returns)
5) Alpha (Excess returns over market)
6) Treynor Ratio (Like Sharpe Ratio but focuses on systematic risk)
7) Sortino Ratio (Like Sharpe and Treynor, but focuses on downside risk)
8) Performance Metrics (Execution time, Memory footprint)

## Project Goals
Maximize accuracy and efficiency by identifying a model with high predictive accuracy while minimizing computational resources, optimize model complexity to ensure robust predictions without overfitting, and enhance portfolio optimization through effective integration of sentiment analysis and machine learning techniques.

## Expected Results
We expect to find the best model among others for portfolio optimization by exploring the impact of sentiment trends and macroeconomic indicators on market movements.

## References

[1] J. C. Van Horne and G. G. C. Parker, "The Random-Walk Theory: An Empirical Test," *Financial Analysts Journal*, vol. 23, no. 6, pp. 87–92, 1967. <br>
[2] L. Lania, R. Collage and M. Vereycken, "The Impact of Uncertainty in Macroeconomic Variables on Stock Returns in the USA," *Journal of Risk Financial Manag. *, vol. 16, no. 3, pp. 189, 2023. [DOI: 10.3390/jrfm16030189](https://doi.org/10.3390/jrfm16030189) <br>
[3] M. Lim, "History of AI Winters," *Actuaries Digital*. Accessed: Feb. 13, 2024. [Online]. Available: [https://www.actuaries.digital/2018/09/05/history-of-ai-winters/](https://www.actuaries.digital/2018/09/05/history-of-ai-winters/)<br>
[4] V. S. Rajput and S. M. Dubey, "Stock market sentiment analysis based on machine learning," in *2016 2nd International Conference on Next Generation Computing Technologies (NGCT)*, Oct. 2016, pp. 506–510. [DOI: 10.1109/NGCT.2016.7877468](https://doi.org/10.1109/NGCT.2016.7877468)<br>
[5] J. Bollen, H. Mao, and X. Zeng, "Twitter mood predicts the stock market," *Journal of Computational Science*, vol. 2, no. 1, pp. 1–8, Mar. 2011. [DOI: 10.1016/j.jocs.2010.12.007](https://doi.org/10.1016/j.jocs.2010.12.007)


## Gantt Chart


<img src= https://github.com/dwu359/ml-portfolio-optimization/assets/141580034/acefaeeb-ce37-45b6-a6d9-ca925db180fd, width = "600">





## Proposal Contribution Table

| Name     | Proposal Contributions                                     |
|----------|-------------------------------------------------------------|
| Sai      | - Worked on the Introduction and Background        |
|          | - Found traditional time-series dataset.       |
| Jungyoun Kwak  | - Found datasets related economic index        |
|          | - Proposed method, results, and discussions.         |
| Member3  | - Contribution.            |
|          | - Contribution.     |
| Member4  | - Contribution.|
|          | - Contribution.    |
| Member5  | - Contribution.|
|          | - Contribution.    |





# ML based Portfolio Management with Macro-Financial Indicators
#### CS7641 Project - Group 11

# I. Introduction

## Literature Review

In rapidly evolving financial markets, traditional investment strategies are increasingly inadequate for risk management and return forecasting [1]. Research has highlighted the importance of macroeconomic influence on stock prices [2]. The lack of data and computing resources constrained the integration of machine learning into investment portfolio management. [3]. However, technological advancements now enable the inclusion of non-financial information such as macroeconomic trends and social media sentiment into predictive models [4][5].

## Dataset
- Federal Interest rate [Link](https://fred.stlouisfed.org/series/DFF)
- GDP  [Link](https://fred.stlouisfed.org/series/GDP)
- CPI  [Link](https://fred.stlouisfed.org/series/CPIAUCSL)
- Personal saving rate [Link](https://fred.stlouisfed.org/series/PSAVERT)
- Unemployment rate [Link](https://fred.stlouisfed.org/series/UNRATE)
- Stock Tweets: This dataset will be utilized for sentiment analysis and potential prediction.[Link](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction)
- News Headlines (RSS Feeds): Tweets covering popular stocks will be analyzed to gauge sentiment.[Link](https://www.kaggle.com/datasets/shtrausslearning/news-trading/data)
- Daily Stock Prices (Yahoo Finance): Daily price data for a subset of S&P 500 stocks since January 2007 via the yfinance API.

# II. Problem Definition

## Problem

The efficacy of traditional portfolio management techniques solely depend on historical stock prices and basic financial indicators. These models fail to account for the broader economic context, resulting in inadequate portfolio allocations and higher risk exposure. To achieve near-perfect forecasts, it is imperative to build more sophisticated frameworks using wider ranges of data sources.


## Motivation:
The motivation behind this project stems from the potential for ML models to address the shortcomings of existing portfolio optimization approaches. By integrating macroeconomic indicators and possibly sentiment analysis into portfolio optimization, we aim to enhance the accuracy and robustness of investment strategies, allowing for better risk management and return maximization and providing investors with valuable insights into market dynamics.

# III. Methods

## 3+ Data Preprocessing Methods:
Collected datasets have different formats and intervals of records. The examples of preprocessing methodologies are listed below:<br>
- Interpolating missing data 
- Removing Outliers
- Data format standardization
- Merge features from multiple datasets
- Analyze and quantize sentiment data

## 3+ ML Algorithms/Models Identified:
We will utilize the supervised/unsupervised models useful for the time series datasets as listed below: (method: explanation)<br>
**Supervised**
1. VAR (Vector Autoregression) - captures relationship between multiple quantities over time
2.  ARIMA (Autoregressive Integrated Moving Average) - predict future trends based on past correlated parameters
3. LSTM - learns order dependence well for longer sequential data without having a vanishing gradient problem 

**Unsupervised**
1. K-Means Clustering - useful for semantic analysis of investor sentiment data.
2. DBSCAN - another semantic clustering technique to compare against K Means
3. SOM (Self-organizing Maps) - Semantic data dimensionality reduction to improve grid clustering


# IV. (Potential) Results and Discussion
## 3+ Quantitative Metrics
1. Sharpe, Treynor, Sortino Ratios ( risk-adjusted return measures)
2. Mean, Std Deviation of Daily Returns
4. Beta (Market Returns Sensitivity)
5. Alpha (Excess returns over market)
6. Execution time
7. Memory footprint

## Project Goals
Maximize accuracy and efficiency of ML techniques while minimizing computational resources, optimize model complexity to ensure robust predictions without overfitting, and enhance portfolio optimization through effective integration of non-financial information.

## Expected Results
We expect to find the best model among others for portfolio optimization by exploring the impact of macroeconomic indicators on market movements.

## References

[1] J. C. Van Horne and G. G. C. Parker, "The Random-Walk Theory: An Empirical Test," *Financial Analysts Journal*, vol. 23, no. 6, pp. 87–92, 1967. <br>
[2] L. Lania, R. Collage and M. Vereycken, "The Impact of Uncertainty in Macroeconomic Variables on Stock Returns in the USA," *Journal of Risk Financial Manag. *, vol. 16, no. 3, pp. 189, 2023. [DOI: 10.3390/jrfm16030189](https://doi.org/10.3390/jrfm16030189) <br>
[3] M. Lim, "History of AI Winters," *Actuaries Digital*. Accessed: Feb. 13, 2024. [Online]. Available: [https://www.actuaries.digital/2018/09/05/history-of-ai-winters/](https://www.actuaries.digital/2018/09/05/history-of-ai-winters/)<br>
[4] V. S. Rajput and S. M. Dubey, "Stock market sentiment analysis based on machine learning," in *2016 2nd International Conference on Next Generation Computing Technologies (NGCT)*, Oct. 2016, pp. 506–510. [DOI: 10.1109/NGCT.2016.7877468](https://doi.org/10.1109/NGCT.2016.7877468)<br>
[5] J. Bollen, H. Mao, and X. Zeng, "Twitter mood predicts the stock market," *Journal of Computational Science*, vol. 2, no. 1, pp. 1–8, Mar. 2011. [DOI: 10.1016/j.jocs.2010.12.007](https://doi.org/10.1016/j.jocs.2010.12.007)


## Gantt Chart


<img src="images/gantt_chart.png" width = "600">





## Proposal Contribution Table

| Name     | Proposal Contributions                                     |
|----------|-------------------------------------------------------------|
| Sai      | - Worked on the Introduction and Background        |
|          | - Found traditional time-series dataset.       |
| Jungyoun Kwak  | - Found datasets related economic index        |
|          | - Proposed method, results, and discussions.         |
| Prabhanjan Nayak  | - Created the proposal presentation,video, and method explanations.            |
|          | - Team management for all proposal deliverables .     |
| Kaushik Arcot  | - Worked on Literature Review for Metrics and Benchmark Performance .|
|          | - Worked on Presentation and Video.    |
| Daniel Wu  | - Proofread and revised proposal |
|          | - Added some metrics    |





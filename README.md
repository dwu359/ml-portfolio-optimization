## CS7641 Project - Group #11
# Predicting Markets with Social Media Sentiment


# I. Introduction

## Literature Review

In the rapidly evolving financial markets, traditional investment strategies like the random walk theory are increasingly inadequate for risk management and return forecasting [1]. Research has highlighted the importance of investor sentiment in stock price predictions, signaling a shift towards more dynamic analytical models [2]. The integration of machine learning into investment portfolio management was constrained until the 1990s by the lack of data and computing resources [3]. However, technological advancements now enable the inclusion of non-financial information such as macroeconomic trends and social media sentiment into predictive models [4][5]. This proposal advocates for the adoption of advanced analytics and machine learning to refine asset allocation, enhance risk management, and uncover deeper market insights. By leveraging comprehensive data analysis, our strategy aims to surpass traditional methods, offering a sophisticated, data-driven approach to investment portfolio management that aligns with contemporary market complexities.

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

While traditional models predominantly rely on financial data, the integration of alternative data streams offers the potential to uncover valuable insights and improve the accuracy of stock price prediction using media sentiment. This project aims to develop a comprehensive approach to portfolio management that takes into account a wider range of market dynamics and improves decision-making processes by investigating the synergies between financial and non-financial data sources.


## Motivation:
A crucial component of finance is managing investment portfolios, which is essential for maximizing profits and reducing risks. Conventional techniques frequently rely on human judgment, which is biased and inefficient. Using machine learning (ML) techniques offers a strong chance to improve decision-making processes by delivering insights based on sentiment data and revealing new directions for portfolio optimization.

# III. Method

## 3+ Data Preprocessing Methods:
Collected datasets have different formats and intervals of records. We will aim to preprocess data to prevent overfitting particular datasets. The examples of preprocessing methodologies are listed below:
- Interpolating missing data 
- Removing Outliers
- Data format standardization
- Anaylze and quantize sentiment data

## 3+ ML Algorithms/Models Identified:
We use stock price, interest rate, GDP, CPI, savings, unemployment rate, and stock sweets in time series. Therefore, we will utilize the supervised/unsupervised models specifically useful for the time series dataset as listed below:
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
[2] M. Baker and J. Wurgler, "Investor Sentiment in the Stock Market," *Journal of Economic Perspectives*, vol. 21, no. 2, pp. 129–151, Apr. 2007. [DOI: 10.1257/jep.21.2.129](https://doi.org/10.1257/jep.21.2.129) <br>
[3] M. Lim, "History of AI Winters," *Actuaries Digital*. Accessed: Feb. 13, 2024. [Online]. Available: [https://www.actuaries.digital/2018/09/05/history-of-ai-winters/](https://www.actuaries.digital/2018/09/05/history-of-ai-winters/)<br>
[4] V. S. Rajput and S. M. Dubey, "Stock market sentiment analysis based on machine learning," in *2016 2nd International Conference on Next Generation Computing Technologies (NGCT)*, Oct. 2016, pp. 506–510. [DOI: 10.1109/NGCT.2016.7877468](https://doi.org/10.1109/NGCT.2016.7877468)<br>
[5] J. Bollen, H. Mao, and X. Zeng, "Twitter mood predicts the stock market," *Journal of Computational Science*, vol. 2, no. 1, pp. 1–8, Mar. 2011. [DOI: 10.1016/j.jocs.2010.12.007](https://doi.org/10.1016/j.jocs.2010.12.007)


## Gantt Chart

![image](https://github.com/dwu359/ml-portfolio-optimization/assets/141580034/09687c0f-bba4-46c3-be7b-384f57292d0e)





## Proposal Contribution Table

| Name     | Proposal Contributions                                     |
|----------|-------------------------------------------------------------|
| Member1  | - Contribution         |
|          | - Contribution.       |
| Jungyoun Kwak  | - Found datasets related economic index        |
|          | - Proposed method, results, and discussions.         |
| Member3  | - Contribution.            |
|          | - Contribution.     |
| Member4  | - Contribution.|
|          | - Contribution.    |
| Member5  | - Contribution.|
|          | - Contribution.    |





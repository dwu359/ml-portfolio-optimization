## Project

# Introduction

## ✅ Literature Review

In the rapidly evolving financial markets, traditional investment strategies like the random walk theory are increasingly inadequate for risk management and return forecasting [1]. Research has highlighted the importance of investor sentiment in stock price predictions, signaling a shift towards more dynamic analytical models [2]. The integration of machine learning into investment portfolio management was constrained until the 1990s by the lack of data and computing resources [3]. However, technological advancements now enable the inclusion of non-financial information such as macroeconomic trends and social media sentiment into predictive models [4][5]. This proposal advocates for the adoption of advanced analytics and machine learning to refine asset allocation, enhance risk management, and uncover deeper market insights. By leveraging comprehensive data analysis, our strategy aims to surpass traditional methods, offering a sophisticated, data-driven approach to investment portfolio management that aligns with contemporary market complexities.

## ✅ Dataset

The dataset comprises daily stock price data obtained from Yahoo Finance via the yfinance API, encompassing a subset of stocks from the S&P 500 index since January 2007. Additionally, money market rates data from the Federal Reserve (FRED) accessed through the quandl API enriches the dataset. Featured stocks include AAPL, GOOG, MSFT, AMZN, INTC, AMD, NVDA, F, TSLA, JPM, MS, and VOO, spanning diverse sectors such as technology, microelectronics, engineering, banking, and finance. The dataset includes features such as Date, Open, High, Low, Close, Adj Close, and Volume.

# II. Problem Statement

## ✅ Problem

While traditional models predominantly rely on financial data, the integration of alternative data streams offers the potential to uncover valuable insights and improve the accuracy of stock price prediction using media sentiment. This project aims to develop a comprehensive approach to portfolio management that takes into account a wider range of market dynamics and improves decision-making processes by investigating the synergies between financial and non-financial data sources.


## ✅ Motivation:
A crucial component of finance is managing investment portfolios, which is essential for maximizing profits and reducing risks. Conventional techniques frequently rely on human judgment, which is biased and inefficient. Using machine learning (ML) techniques offers a strong chance to improve decision-making processes by delivering insights based on sentiment data and revealing new directions for portfolio optimization.

# III. Method

## ✅ 3+ Data Preprocessing Methods:
- Anaylze and quantize sentiment data
- Interpolating missing data 
- Removing Outliers
- data format

## ✅ 3+ ML Algorithms/Models Identified:
**Supervised**
1) VAR (Vector Autoregression)
2) ARIMA (Autoregressive Integrated Moving Average)
3) LSTM

**Unsupervised**
1) K-Means Clustering
2) DBSCAN
3) SOM (Self-organizing Maps)

# IV. (Potential) Results and Discussion
## ✅3+ Quantitative Metrics
1)	Accuracy
2)	Epoch
3)	Runtime
4)	PCA

## ✅Project Goals
Find out the most accurate and efficient model for portfolio prediction based on sentiment analysis.

## ✅Expected Results
We expect to find the best model among others for portfolio optimization.

## References

[1] J. C. Van Horne and G. G. C. Parker, "The Random-Walk Theory: An Empirical Test," *Financial Analysts Journal*, vol. 23, no. 6, pp. 87–92, 1967.
[2] M. Baker and J. Wurgler, "Investor Sentiment in the Stock Market," *Journal of Economic Perspectives*, vol. 21, no. 2, pp. 129–151, Apr. 2007. [DOI: 10.1257/jep.21.2.129](https://doi.org/10.1257/jep.21.2.129) <br>
[3] M. Lim, "History of AI Winters," *Actuaries Digital*. Accessed: Feb. 13, 2024. [Online]. Available: [https://www.actuaries.digital/2018/09/05/history-of-ai-winters/](https://www.actuaries.digital/2018/09/05/history-of-ai-winters/)<br>
[4] V. S. Rajput and S. M. Dubey, "Stock market sentiment analysis based on machine learning," in *2016 2nd International Conference on Next Generation Computing Technologies (NGCT)*, Oct. 2016, pp. 506–510. [DOI: 10.1109/NGCT.2016.7877468](https://doi.org/10.1109/NGCT.2016.7877468)<br>
[5] J. Bollen, H. Mao, and X. Zeng, "Twitter mood predicts the stock market," *Journal of Computational Science*, vol. 2, no. 1, pp. 1–8, Mar. 2011. [DOI: 10.1016/j.jocs.2010.12.007](https://doi.org/10.1016/j.jocs.2010.12.007)




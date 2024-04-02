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

Collected datasets have different formats and intervals of records. The examples of preprocessing methodologies are listed below:<br>
- Interpolating missing data 
- Removing Outliers
- Data format standardization
- Merge features from multiple datasets
- Analyze and quantize sentiment data

We utilized supervised/unsupervised models useful for the time series datasets.
Our current models include K-means clustering for unsupervised and
Autoregressive Integrated Moving Average (ARIMA) for supervised.

We chose K-means particularly because it is simple and fast, and we did not feel
the need to obtain probabilities from a soft clustering model like GMM.  We plan
to use K-means primarily to analyse stock similarity, facilitating the selection
of stocks across various clusters to enhance portfolio diversification. Some
challenges that using a clustering algorithm like K-means poses for this,
however, is determining a proper distance function that accurately identifies
similarities between stocks.

We primarily utilized sklearn to train our K-means and PCA models and for
performance metrics for clustering, and we also utilized pandas and numpy for
general data manipulation. 

To simplify data preprocessing, we limited our model to 2 features and
experimented with what we chose for the features, such as mean returns,
volatility, GDP growth, unemployment rate, etc. We mainly compared stocks via
mean returns and volatility, which is calculated via the mean and variance of
each stock’s percent change in value over the last 5 years. We chose mean
returns and volatility primarily because they were simple stock-specific
indicators. We also attempted to compare stocks via macro-economic indicators
such as GDP growth, unemployment rate, and CPI.

We also compared the performance of our chosen features to feature extraction
via PCA. We wanted to compare the manually selected features we chose to
machine learning methods of selecting features. We chose PCA specifically
because PCA is a popular unsupervised learning method of performing feature
extraction.

 <img src="images/kmeans_images/kmeans_wo_pca.png" width = "600">
 <img src="images/kmeans_images/kmeans_pca.png" width = "600">

We explored using K-means to cluster 10 of most popular stock indices. Although
K-means doesn’t select stocks directly, we altered the algorithm to choose the
stocks with the minimum volatility for each cluster. We estimated our K-values
by seeing which datapoints on the graphs were closest to each other. Our K-means
algorithm for mean returns vs volatility for k=3 selected JPM, NVDA, and VOO,
and our K-means algorithm with PCA for k=4 selected JPM, NVDA, TSLA, and VOO.
Since PCA turned out to output quite similar results compared to mean returns vs
volatility, this confirmed our assumptions that mean returns and volatility were
good measurements to categorize stocks.

<img src="images/kmeans_images/kmeans_elbow_method.png" width = "600">

Through the elbow method, we confirmed that the most optimal number of clusters
without PCA was 3, and the optimal number of clusters with PCA was 4.

For the supervised learning aspect of our project, we opted for the ARIMA (Autoregressive Integrated Moving Average) model, renowned as one of the most widely used models for forecasting linear time series data. Its extensive adoption stems from its reliability, efficiency, and capability in forecasting short-term fluctuations in stock market movements.
ARIMA models are a popular tool in time series analysis, particularly in forecasting future values based on historical data. By incorporating lagged moving averages, ARIMA models provide a method to smooth out fluctuations and identify patterns in time series data. However, their reliance on past observations inherently assumes that future behavior will resemble the past, making them susceptible to inaccuracies in volatile market conditions like financial crises or during periods of rapid technological advancements. While ARIMA models offer valuable insights, it's essential to supplement their predictions with other analytical techniques and consider the broader economic and technological landscape for more robust forecasting in dynamic markets.
We conducted exploratory data analysis (EDA) on a selection of 10 ticker stocks, focusing on their adjusted closing prices from 2018 to 2024. Our EDA encompassed various aspects, including the visualization of stock volatility, daily returns, quarterly returns, annual returns, adjusted closing prices over the years, and a covariance matrix heatmap of portfolio returns. 

<img src="images/volitality_plots/adj_closed_all.png" width = "600">

<img src="images/volitality_plots/annual_returns.png" width = "600">

<img src="images/volitality_plots/quarterly_returns.png" width = "600">

<img src="images/volitality_plots/cov_heatmap.png" width = "600">

<img src="images/volitality_plots/portfolio_volitality.png" width = "600">


While preparing our data for modeling, we rigorously checked for seasonality and
stationarity, both crucial conditions for employing the ARIMA model using the
python statsmodel library. Seasonality, characterized by regular and predictable
patterns that repeat over a calendar year, can adversely impact regression
models. To ensure stationarity, a prerequisite for ARIMA, we employed
differencing techniques to remove trends and seasonal structures from the data.
The Augmented Dickey-Fuller (ADF) test emerged as a pivotal tool in our
analysis, widely employed in time series analysis for assessing stationarity [6].
This statistical test aims to determine the presence of a unit root within a
series, indicating non-stationarity. The test formulates null and alternative
hypotheses, with the null hypothesis positing the existence of a unit root, and
the alternative hypothesis suggesting its absence. Failure to reject the null
hypothesis indicates non-stationarity, implying the series may exhibit either
linear or difference stationary characteristics.

<img src="images/arima_images/seasonal_decompose.png" width = "600">

In addition to checking for seasonality and stationarity, we performed seasonal decomposition to extract the underlying time series components, namely Trend and Seasonality, from our data. To mitigate the magnitude of values and address any growing trends within the series, we initially applied a logarithmic transformation. Subsequently, we calculated the rolling average of the log-transformed series, aggregating data from the preceding 12 months to compute mean consumption values at each subsequent point in the series. Following data preprocessing, we partitioned the dataset into training and testing subsets, preserving the chronological order of observations to maintain the time-series nature of the data.
For tuning the hyperparameters of the ARIMA model, denoted as a tuple (p, d, q),
we employed a systematic approach. Here, 'p' represents the number of
autoregressive terms, 'd' denotes the number of nonseasonal differences, and 'q'
signifies the number of lagged forecast errors in the prediction equation. To
determine suitable values for 'p' and 'q', we analysed autocorrelation and
partial autocorrelation plots. Specifically, we utilized the partial
autocorrelation function (PACF) plot to identify the value of 'p', as the cutoff
point in the PACF plot indicates the autoregressive term. Conversely, the
autocorrelation function (ACF) plot aided in determining the value of 'q', with
the cutoff point in the ACF plot corresponding to the moving average term [6].
Additionally, we leveraged grid search methods and the auto_arima function to systematically explore various combinations of hyperparameters and select the optimal configuration for our ARIMA model. This comprehensive approach ensured robust parameter tuning, facilitating the development of an effective forecasting model for our time series data.


# IV. (Potential) Results and Discussion

## Project Goals
Maximize accuracy and efficiency of ML techniques while minimizing computational resources, optimize model complexity to ensure robust predictions without overfitting, and enhance portfolio optimization through effective integration of non-financial information.

<img src="images/kmeans_images/kmeans_metrics.png" width = "600">

Ultimately, we found that feature extraction via PCA performed slightly better than mean returns vs volatility in Silhouette, Davies Bouldin, and Calinski Harabasz scores for K values from 3 to 9. We chose these metrics because all three of these didn’t require ground truth labels. Silhouette most matches our observation that k=3 is the best choice for K. However, Davies Bouldin and Calinski Harabasz metrics suggest that k=9 is better especially for PCA, which contradicts what we have seen through the elbow method. Therefore, we suspect that PCA may be overfitted since the top principal components are not as well defined, so they may capture irrelevant details in the dataset. 
Overall, we conclude the current K-means algorithm with and without PCA is an
average performer since it is able to reach a value slightly above 0.5 for
Silhouette score for k=3 and k=4, and a Davies Bouldin score below 0.4.
Additionally, the low Calinski Harabasz score for k=3 may show that k=3 is not
as overfitted as k=9.

<img src="images/arima_images/acf_pcf.png" width = "600">

For the ARIMA model, based on the partial autocorrelation (PACF) and autocorrelation (ACF) plots, along with grid search methodology, we determined the optimal parameters for the ARIMA model to be (p, d, q) = (1,1,0).

<img src="images/arima_images/rolling_mean_sd.png" width = "600">

Furthermore, examination of the plot for rolling mean and standard deviation
revealed an increasing trend in both metrics. Additionally, with a p-value
exceeding 0.05 and the test statistic surpassing the critical values, indicating
non-stationarity, the time series data is deemed to be non-linear. Consequently,
to address this non-linearity, the natural logarithm of the time series was
applied, as shown below. 

<img src="images/arima_images/log_roll_mean.png" width = "600">

As observed, the application of log-transformation to the time series has
induced a slight linearity, rendering it amenable to modelling. This
transformation has effectively mitigated the non-linear trends present in the
original data.



## Next Steps

For K-means, we realized that our choice of macro-economic indicators such as GDP, CPI, unemployment rate may not be the best candidates for features for unsupervised clustering since these economic indicators don’t differentiate between stocks. In the future, we plan to utilize more stock specific economic indicators to increase the performance of our clustering algorithm.

For the K-means model without PCA (k=3), and for the KMeans model with PCA
(k=4), along with the ARIMA model for Portfolio optimization, here is a table
comparing the metrics vs the benchmark model, implementing the maximization of
Sharpe ratio using modern portfolio theory.

| Model | Mean Daily Returns | Std Dev of Daily Returns | Sharpe Ratio |
Treynor Ratio | Beta | Alpha | Cumulative Return |

|----------|----------------|---------------------|---------------------|-------|---------|-------------|-------------|
| Benchmark | 0.0037 | 0.0267 | 2.205 | 0.00167 | 2.103 | 0.0016 | 1.301 |
| K Means (w/o PCA) | 0.00243 | 0.0148 | 2.609 | 0.00165 | 1.351 | 0.0011 | 0.782 |
| Kmeans (w PCA) | 0.00262 | 0.0161 | 2.588 | 0.00160 | 1.514 | 0.0011 | 0.859 |
| ARIMA + MPT | 0.00223 | 0.0187 | 1.937 | 0.00141 | 1.478 | 0.0008 | 0.688 |



Given the limitations encountered with the ARIMA model, we intend to leverage its outputs as input features for a more sophisticated model, such as a Recurrent Neural Network (RNN) or Long Short-Term Memory (LSTM) network. These architectures have demonstrated remarkable efficacy in capturing complex temporal dependencies, making them particularly suitable for modeling economic indicators. By integrating ARIMA predictions alongside other relevant micro-economic indicators into the neural network, we anticipate achieving improved forecasting accuracy and robustness. 



## References

[1] J. C. Van Horne and G. G. C. Parker, "The Random-Walk Theory: An Empirical Test," *Financial Analysts Journal*, vol. 23, no. 6, pp. 87–92, 1967. <br>
[2] L. Lania, R. Collage and M. Vereycken, "The Impact of Uncertainty in Macroeconomic Variables on Stock Returns in the USA," *Journal of Risk Financial Manag. *, vol. 16, no. 3, pp. 189, 2023. [DOI: 10.3390/jrfm16030189](https://doi.org/10.3390/jrfm16030189) <br>
[3] M. Lim, "History of AI Winters," *Actuaries Digital*. Accessed: Feb. 13, 2024. [Online]. Available: [https://www.actuaries.digital/2018/09/05/history-of-ai-winters/](https://www.actuaries.digital/2018/09/05/history-of-ai-winters/)<br>
[4] V. S. Rajput and S. M. Dubey, "Stock market sentiment analysis based on machine learning," in *2016 2nd International Conference on Next Generation Computing Technologies (NGCT)*, Oct. 2016, pp. 506–510. [DOI: 10.1109/NGCT.2016.7877468](https://doi.org/10.1109/NGCT.2016.7877468)<br>
[5] J. Bollen, H. Mao, and X. Zeng, "Twitter mood predicts the stock market,"
*Journal of Computational Science*, vol. 2, no. 1, pp. 1–8, Mar. 2011. [DOI:
10.1016/j.jocs.2010.12.007](https://doi.org/10.1016/j.jocs.2010.12.007)<br>
[6] Hayes, A. (2024, February 23). Autoregressive integrated moving average (ARIMA) prediction model. Investopedia. https://www.investopedia.com/terms/a/autoregressive-integrated-moving-average-arima.asp 


## Gantt Chart


<img src="images/project_mgmt/gantt_chart.png" width = "600">

## Proposal Contribution Table

| Name     | Proposal Contributions                                     |
|----------|-------------------------------------------------------------|
| Sai      | - Combining stock dataset and EDA        |
|          | - Data pre-processing and visualizations for ARIMA.       |
| Jungyoun Kwak  | - Collect economic indicators data set and pre-processing data        |
|          | - Implementing code and visualizations for Kmeans.         |
| Prabhanjan Nayak  | - Finalized models, repository structure and description, and report.            |
|          | - Team management for all midterm deliverables.     |
| Kaushik Arcot  | - Compiled code for modelling ARIMA for all stocks , and code for Benchmark Model using Modern Portfolio Theory|
|          | - Completed code and compiled metrics of analyzing portfolio allocations. |
| Daniel Wu  | - PCA feature reduction for Kmeans |
|          | - Wrote the methods and results for Kmeans    |





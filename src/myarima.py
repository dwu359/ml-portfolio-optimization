import matplotlib.pyplot as plt
import numpy as np
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


class MyARIMA:
    def __init__(self, time_series_data):
        self.time_series_data = time_series_data

    def check_stationarity(self, data):
        """
        Checks the stationarity of a time series by calculating the rolling mean and standard deviation,
        plotting the data and rolling statistics, and performing the Augmented Dickey-Fuller (ADF) test.

        Returns:
            None
        """
        # Calculate rolling mean and standard deviation
        rolling_mean = data.rolling(window=12).mean()
        rolling_std = data.rolling(window=12).std()

        # Plot the data and rolling statistics
        plt.figure(figsize=(10, 6))
        plt.plot(data, color="blue", label="Original Data")
        plt.plot(rolling_mean, color="red", label="Rolling Mean")
        plt.plot(rolling_std, color="green", label="Rolling Std")
        plt.legend(loc="best")
        plt.title("Rolling Mean and Standard Deviation")
        plt.show()

        # Perform ADF test
        adf_result = adfuller(data)

        # Print ADF test results
        print("ADF Statistic:", adf_result[0])
        print("p-value:", adf_result[1])
        print("Critical Values:")
        for key, value in adf_result[4].items():
            print(f"{key}: {value}")

    def log_data(self, data):
        """
        Takes the log of the time series data and plots the rolling mean and standard deviation.

        Returns:
            None
        """
        # Take the log of the time series data
        log_data = np.log(data)
        return log_data

    def difference_data(self, log_data, shift=1):
        """
        Takes the difference of the log data and plots the rolling mean and standard deviation.

        Args:
            log_data: Log of the time series data
            shift : Window of Difference
        """
        # Take the difference of the log data
        diff_data = log_data - log_data.shift(shift)

        # Calculate rolling mean of the difference data
        rolling_mean = diff_data.rolling(window=12).mean()
        rolling_std = diff_data.rolling(window=12).std()

        return diff_data

    def acf_pacf_plots(self, diff_data):
        """
        Plots the ACF and PACF for the difference data.

        Args:
            diff_data: Difference of the log data
        """
        # Plot ACF and PACF
        fig, ax = plt.subplots(2, figsize=(12, 8))
        plot_acf(diff_data.dropna(), ax=ax[0], lags=20)
        plot_pacf(diff_data.dropna(), ax=ax[1], lags=20)
        plt.show()

    def fit_arima_parameters(self, data):
        """
        Fit (p,d,q) of the ARIMA model for stock data
        Args:
            data : stock data to model
        """
        model = auto_arima(data, seasonal=True, m=4, maxiter=500)
        self.order = model.order
        self.seasonal_order = model.seasonal_order

    def forecast(self, data, N):
        """
        Forecast Prices for the test period using the fitted ARIMA model
        """
        arima_model = ARIMA(
            data.values, order=self.order, seasonal_order=self.seasonal_order
        )
        arima_fit = arima_model.fit()
        self.model = arima_model
        forecast = arima_fit.forecast(steps=N)
        return forecast

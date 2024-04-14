import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

class mydbscan:
    def __init__(self):
        self.stock_files = {
            'NVDA': 'NVDA_data.csv',
            'VOO': 'VOO_data.csv',
            'JPM': 'JPM_data.csv',
            'MS': 'MS_data.csv',
            'TSLA': 'TSLA_data.csv',
            'AMD': 'AMD_data.csv',
            'F': 'F_data.csv',
            'AMZN': 'AMZN_data.csv',
            'GOOG': 'GOOG_data.csv',
            'INTC': 'INTC_data.csv'
        }
        self.balance_sheet_path = "./dataset/stock_balance_sheet.csv"
        self.balance_sheet_data = pd.read_csv(self.balance_sheet_path, parse_dates=["DATE"], index_col="DATE")
        self.balance_sheet_quarterly = self.balance_sheet_data.resample("Q").last()
        self.quarterly_returns_all_stocks = pd.DataFrame()
        self.financial_metrics = [
            "Revenue Growth (YoY)", "Shares Change", "Gross Margin", "Operating Margin",
            "Profit Margin", "Free Cash Flow Margin", "EBITDA Margin", "EBIT Margin",
            "Cash Growth", "Debt Growth"
        ]
        self.scaler = StandardScaler()

    def load_and_process_stock_data(self):
        for symbol, filename in self.stock_files.items():
            stock_data_path = f"./dataset/stock_data/{filename}"
            stock_data = pd.read_csv(stock_data_path, parse_dates=["Date"], index_col="Date")
            daily_returns = stock_data["Close"].pct_change().dropna()
            quarterly_returns = daily_returns.resample("Q").mean()
            self.quarterly_returns_all_stocks[symbol] = quarterly_returns

        # Ensure there's no missing data across stocks
        self.quarterly_returns_all_stocks.dropna(inplace=True)

    def perform_clustering_with_balance(self):
        for metric in self.financial_metrics:
            features = pd.DataFrame({
                "Mean_Quarterly_Returns": self.quarterly_returns_all_stocks.mean(axis=1),
                metric: self.balance_sheet_quarterly[metric].reindex(self.quarterly_returns_all_stocks.index).ffill(),
            }).dropna()

            features_scaled = self.scaler.fit_transform(features)
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            clusters = dbscan.fit_predict(features_scaled)

            # Evaluation metrics
            silhouette = silhouette_score(features_scaled, clusters)
            davies_bouldin = davies_bouldin_score(features_scaled, clusters)
            calinski_harabasz = calinski_harabasz_score(features_scaled, clusters)

            # Plotting
            sns.scatterplot(x=features['Mean_Quarterly_Returns'], y=features[metric].astype(float), hue=clusters, palette='viridis')
            plt.title(f'Clusters for {metric}')
            plt.xlabel('Mean Quarterly Returns')
            plt.ylabel(metric)
            plt.show()

            print(f"Metrics for {metric} - Silhouette: {silhouette}, Davies Bouldin: {davies_bouldin}, Calinski Harabasz: {calinski_harabasz}")



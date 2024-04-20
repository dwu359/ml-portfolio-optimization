import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestNeighbors

class MyDBSCAN:
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

        for symbol, filename in self.stock_files.items():
            stock_data_path = f"./dataset/stock_data/{filename}"
            stock_data = pd.read_csv(stock_data_path, parse_dates=["Date"], index_col="Date")
            daily_returns = stock_data["Close"].pct_change().dropna()
            quarterly_returns = daily_returns.resample("Q").mean()
            self.quarterly_returns_all_stocks[symbol] = quarterly_returns

        # Ensure there's no missing data across stocks
        self.quarterly_returns_all_stocks.dropna(inplace=True)

        self.financial_metrics = [
            "Revenue Growth (YoY)", "Shares Change", "Gross Margin", "Operating Margin",
            "Profit Margin", "Free Cash Flow Margin", "EBITDA Margin", "EBIT Margin",
            "Cash Growth", "Debt Growth"
        ]

        self.features = None
        self.features_scaled = None
        self.data = None
        self.scaler = StandardScaler()

        self.metrics_without_pca = {
            "k": [],
            "silhouette": [],
            "davies_bouldin": [],
            "calinski_harabasz": [],
        }
        self.metrics_with_pca = {
            "k": [],
            "silhouette": [],
            "davies_bouldin": [],
            "calinski_harabasz": [],
        }
    
    def set_stock_data(self):
        # Load and preprocess data
        stock_symbols = list(self.stock_files.keys())
        stock_data = {}

        for symbol in stock_symbols:
            stock_data[symbol] = pd.read_csv(
                "./dataset/stock_data/" + symbol + "_data.csv", usecols=["Date", "Close"]
            )
            stock_data[symbol]["Date"] = pd.to_datetime(stock_data[symbol]["Date"])
            stock_data[symbol].set_index("Date", inplace=True)

        # Align dataframes to the same date range and fill missing values
        df_close = pd.DataFrame(
            {symbol: data["Close"] for symbol, data in stock_data.items()}
        )
        df_close = df_close.fillna(method="ffill").dropna()

        # Feature extraction
        self.data = df_close.pct_change().dropna()

    def clustering_with_quarterly_returns(self):
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
    
    def volatility_feature_engineering(self, pca_flag= False):
        if not pca_flag:
            self.features = pd.DataFrame()
            self.features["mean_returns"] = self.data.mean()
            self.features["volatility"] = self.data.std()
            self.features_scaled = self.scaler.fit_transform(self.features)
        else:
            pca = PCA(n_components=2)
            daily_returns_t = self.data.T
            pca.fit(daily_returns_t)
            daily_returns_pca = pca.transform(daily_returns_t)
            self.features = pd.DataFrame(daily_returns_pca, index=self.data.columns, columns=['PC1', 'PC2'])
            self.features_scaled = self.scaler.fit_transform(self.features)
             
    def get_elbow_plot(self):
        #Elbow plot
        tree = BallTree(self.features_scaled, leaf_size=2)
        k = 4
        dist, _ = tree.query(self.features_scaled, k=k)
        plt.plot(np.arange(self.features_scaled.shape[0]), np.sort(dist[:, k-1]), linewidth=2, markersize=5)
        plt.xlabel("Points sorted by Kth Nearest Neighbor")
        plt.ylabel("Kth Nearest Neighbor Distance")
        plt.title("DBSCAN Elbow Method for Optimal Eps")
        plt.grid(True)
        plt.show()

    def volatility_clustering_without_pca(self, range_n_clusters=None):
        #Clustering
        print("Clustering without PCA")
        range_n_clusters = np.arange(1.25, 1.75, 0.25)
        for eps in range_n_clusters:
            dbscan = DBSCAN(eps=eps, min_samples=3)
            self.features['cluster'] = dbscan.fit_predict(self.features_scaled)
            # Evaluation metrics
            silhouette_avg = silhouette_score(self.features_scaled, self.features['cluster'])
            davies_bouldin = davies_bouldin_score(self.features_scaled, self.features['cluster'])
            calinski_harabasz = calinski_harabasz_score(self.features_scaled, self.features['cluster'])

            # Storing metrics
            self.metrics_without_pca['eps'].append(eps)
            self.metrics_without_pca['silhouette'].append(silhouette_avg)
            self.metrics_without_pca['davies_bouldin'].append(davies_bouldin)
            self.metrics_without_pca['calinski_harabasz'].append(calinski_harabasz)

            selected_stocks = self.features.groupby('cluster')['volatility'].idxmin().values.tolist()
            print(f"Suggested Stocks without PCA for eps={eps}: {selected_stocks}")
            print("Silhouette", self.metrics_without_pca['silhouette'][-1])
            print("Davies Bouldin", self.metrics_without_pca['davies_bouldin'][-1])
            print("Calinski Harabasz", self.metrics_without_pca['calinski_harabasz'][-1])


            # Plotting
            sns.scatterplot(data=self.features, x='mean_returns', y='volatility', hue='cluster', palette='viridis', s=100)
            plt.title(f'Stock Clusters without PCA (eps={eps})')
            plt.xlabel('Mean Returns')
            plt.ylabel('Volatility')
            plt.legend()
            plt.show()

    def volatility_clustering_with_pca(self, range_n_clusters=None):
        #Clustering
        print("Clustering with PCA")
        range_n_clusters = np.arange(0.75, 1.25, 0.25)
        for eps in range_n_clusters:
            dbscan = DBSCAN(eps=eps, min_samples=3)
            self.features['cluster'] = dbscan.fit_predict(self.features_scaled)
            # Evaluation metrics
            silhouette_avg = silhouette_score(self.features_scaled, self.features['cluster'])
            davies_bouldin = davies_bouldin_score(self.features_scaled, self.features['cluster'])
            calinski_harabasz = calinski_harabasz_score(self.features_scaled, self.features['cluster'])

            # Storing metrics
            self.metrics_with_pca['eps'].append(eps)
            self.metrics_with_pca['silhouette'].append(silhouette_avg)
            self.metrics_with_pca['davies_bouldin'].append(davies_bouldin)
            self.metrics_with_pca['calinski_harabasz'].append(calinski_harabasz)

            selected_stocks = self.features.groupby('cluster')['volatility'].idxmin().values.tolist()
            print(f"Suggested Stocks with PCA for eps={eps}: {selected_stocks}")
            print("Silhouette", self.metrics_with_pca['silhouette'][-1])
            print("Davies Bouldin", self.metrics_with_pca['davies_bouldin'][-1])
            print("Calinski Harabasz", self.metrics_with_pca['calinski_harabasz'][-1])


            # Plotting
            sns.scatterplot(data=self.features, x='mean_returns', y='volatility', hue='cluster', palette='viridis', s=100)
            plt.title(f'Stock Clusters with PCA (eps={eps})')
            plt.xlabel('Mean Returns')
            plt.ylabel('Volatility')
            plt.legend()
            plt.show()

    def get_nearest_neighbors(self):
        # Assuming features_scaled is your dataset
        neighbors = NearestNeighbors(n_neighbors=4)  # Use n_neighbors = MinPts
        neighbors_fit = neighbors.fit(self.features_scaled)
        distances, indices = neighbors_fit.kneighbors(self.features_scaled)

        # Sort distance values by ascending value and plot
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]  # Second closest point
        plt.plot(distances)
        plt.title('K-Nearest Neighbors Distances')
        plt.xlabel('Points sorted by distance')
        plt.ylabel('Distance to nearest point')
        plt.show()

    
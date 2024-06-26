import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


class MyKMeans:

    def __init__(self):
        self.stock_files = {
            "NVDA": "NVDA_data.csv",
            "VOO": "VOO_data.csv",
            "JPM": "JPM_data.csv",
            "MS": "MS_data.csv",
            "TSLA": "TSLA_data.csv",
            "AMD": "AMD_data.csv",
            "F": "F_data.csv",
            "AMZN": "AMZN_data.csv",
            "GOOG": "GOOG_data.csv",
            "INTC": "INTC_data.csv",
        }

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

    def set_econ_data(self):
        # Initialize DataFrame for monthly returns
        self.data = pd.DataFrame()

        # Process each stock file for monthly returns
        for symbol, filename in self.stock_files.items():
            stock_data_path = f"./dataset/stock_data/{filename}"
            stock_data = pd.read_csv(
                stock_data_path, parse_dates=["Date"], index_col="Date"
            )
            daily_returns = stock_data["Close"].pct_change().dropna()
            monthly_returns = daily_returns.resample("M").mean()
            self.data[symbol] = monthly_returns

        # Ensure there's no missing data across stocks
        self.data.dropna(inplace=True)



    def volatility_feature_engineering(self):
        self.features = pd.DataFrame()
        self.features["mean_returns"] = self.data.mean()
        self.features["volatility"] = self.data.std()

        
        self.features_scaled = self.scaler.fit_transform(self.features)

    def volatility_clustering_without_pca(self, range_n_clusters):
        for k in range_n_clusters:
            kmeans = KMeans(n_clusters=k, random_state=42)
            self.features["cluster"] = kmeans.fit_predict(self.features_scaled)

            # Evaluation metrics
            silhouette_avg = silhouette_score(
                self.features_scaled, self.features["cluster"]
            )
            davies_bouldin = davies_bouldin_score(
                self.features_scaled, self.features["cluster"]
            )
            calinski_harabasz = calinski_harabasz_score(
                self.features_scaled, self.features["cluster"]
            )
            
            print(f"\nFor k={k}:")
            print(f"Silhouette Score: {silhouette_avg :.4f}")
            print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
            print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
            # Storing metrics
            self.metrics_without_pca["k"].append(k)
            self.metrics_without_pca["silhouette"].append(silhouette_avg)
            self.metrics_without_pca["davies_bouldin"].append(davies_bouldin)
            self.metrics_without_pca["calinski_harabasz"].append(calinski_harabasz)

            selected_stocks = (
                self.features.groupby("cluster")["volatility"].idxmin().values.tolist()
            )
            print(f"Suggested Stocks without PCA for k={k}: {selected_stocks}")

            # Plotting
            sns.scatterplot(
                data=self.features,
                x="mean_returns",
                y="volatility",
                hue="cluster",
                palette="viridis",
                s=100,
            )
            plt.title(f"Stock Clusters without PCA (k={k})")
            plt.xlabel("Mean Returns")
            plt.ylabel("Volatility")
            plt.legend()
            plt.show()
    
    def volatility_clustering_with_pca(self, range_n_clusters, n_components=2):
        # First, ensure that the features for PCA (mean_returns, volatility) are scaled
        self.volatility_feature_engineering()  # This prepares self.features_scaled

        # Perform PCA transformation on scaled features
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(self.features_scaled)

        # Create a DataFrame for PCA components
        pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2'], index=self.features.index)

        # Re-add the 'volatility' column to the new DataFrame with PCA components
        pca_df['volatility'] = self.features['volatility']

        # Update self.features to include PCA components AND the 'volatility' column
        self.features = pca_df

        # Scale PCA components for clustering
        self.features_scaled = self.scaler.fit_transform(self.features[['PC1', 'PC2']])

        print("\nClustering with PCA")
        for k in range_n_clusters:
            if k == 0:  # KMeans cannot cluster with 0 clusters
                continue

            # Clustering with PCA-transformed and scaled features
            kmeans_pca = KMeans(n_clusters=k, random_state=42)
            self.features['cluster'] = kmeans_pca.fit_predict(self.features_scaled)

            # Evaluation metrics for clustering
            silhouette_avg_pca = silhouette_score(self.features_scaled, self.features['cluster'])
            davies_bouldin_pca = davies_bouldin_score(self.features_scaled, self.features['cluster'])
            calinski_harabasz_pca = calinski_harabasz_score(self.features_scaled, self.features['cluster'])
            
            print(f"\nFor k={k}:")
            print(f"Silhouette Score: {silhouette_avg_pca :.4f}")
            print(f"Davies-Bouldin Index: {davies_bouldin_pca:.4f}")
            print(f"Calinski-Harabasz Index: {calinski_harabasz_pca:.4f}")
            
            # Store the evaluation metrics
            self.metrics_with_pca['k'].append(k)
            self.metrics_with_pca['silhouette'].append(silhouette_avg_pca)
            self.metrics_with_pca['davies_bouldin'].append(davies_bouldin_pca)
            self.metrics_with_pca['calinski_harabasz'].append(calinski_harabasz_pca)

            # Select the least volatile stock in each cluster
            selected_stocks_pca = self.features.groupby('cluster')['volatility'].idxmin().values.tolist()
            print(f"Suggested Stocks with PCA for k={k}: {selected_stocks_pca}")

            # Plotting the clusters
            sns.scatterplot(data=self.features, x='PC1', y='PC2', hue='cluster', palette='viridis', s=100)
            plt.title(f'Stock Clusters with PCA (k={k})')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend()
            plt.show()

    
    def plot_metrics_pca(self):
        # Plotting evaluation metrics for Clustering with PCA
        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        axs[0].plot(self.metrics_with_pca['k'], self.metrics_with_pca['silhouette'], marker='o', linestyle='-', color='blue', label='Silhouette Score')
        axs[0].set_title('Silhouette Score vs. Number of Clusters with PCA')
        axs[0].set_xlabel('Number of Clusters')
        axs[0].set_ylabel('Silhouette Score')
        axs[0].grid(True)

        axs[1].plot(self.metrics_with_pca['k'], self.metrics_with_pca['davies_bouldin'], marker='o', linestyle='-', color='red', label='Davies-Bouldin Index')
        axs[1].set_title('Davies-Bouldin Index vs. Number of Clusters with PCA')
        axs[1].set_xlabel('Number of Clusters')
        axs[1].set_ylabel('Davies-Bouldin Index')
        axs[1].grid(True)

        axs[2].plot(self.metrics_with_pca['k'], self.metrics_with_pca['calinski_harabasz'], marker='o', linestyle='-', color='green', label='Calinski-Harabasz Index')
        axs[2].set_title('Calinski-Harabasz Index vs. Number of Clusters with PCA')
        axs[2].set_xlabel('Number of Clusters')
        axs[2].set_ylabel('Calinski-Harabasz Score')
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_metrics_without_pca(self):
        # Plotting evaluation metrics for Clustering without PCA
        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        axs[0].plot(self.metrics_without_pca['k'], self.metrics_without_pca['silhouette'], marker='o', linestyle='-', color='blue', label='Silhouette Score')
        axs[0].set_title('Silhouette Score vs. Number of Clusters without PCA')
        axs[0].set_xlabel('Number of Clusters')
        axs[0].set_ylabel('Silhouette Score')
        axs[0].grid(True)

        axs[1].plot(self.metrics_without_pca['k'], self.metrics_without_pca['davies_bouldin'], marker='o', linestyle='-', color='red', label='Davies-Bouldin Index')
        axs[1].set_title('Davies-Bouldin Index vs. Number of Clusters without PCA')
        axs[1].set_xlabel('Number of Clusters')
        axs[1].set_ylabel('Davies-Bouldin Index')
        axs[1].grid(True)

        axs[2].plot(self.metrics_without_pca['k'], self.metrics_without_pca['calinski_harabasz'], marker='o', linestyle='-', color='green', label='Calinski-Harabasz Index')
        axs[2].set_title('Calinski-Harabasz Index vs. Number of Clusters without PCA')
        axs[2].set_xlabel('Number of Clusters')
        axs[2].set_ylabel('Calinski-Harabasz Score')
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_metrics(self):
        # Plotting Silhouette Scores
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_without_pca['k'], self.metrics_without_pca['silhouette'], label='Without PCA', marker='o')
        plt.plot(self.metrics_with_pca['k'], self.metrics_with_pca['silhouette'], label='With PCA', marker='x')
        plt.title('Silhouette Score Comparison')
        plt.xlabel('Number of Clusters k')
        plt.ylabel('Silhouette Score')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plotting Davies-Bouldin Indexes
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_without_pca['k'], self.metrics_without_pca['davies_bouldin'], label='Without PCA', marker='o')
        plt.plot(self.metrics_with_pca['k'], self.metrics_with_pca['davies_bouldin'], label='With PCA', marker='x')
        plt.title('Davies-Bouldin Index Comparison')
        plt.xlabel('Number of Clusters k')
        plt.ylabel('Davies-Bouldin Index')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plotting Calinski-Harabasz Scores
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_without_pca['k'], self.metrics_without_pca['calinski_harabasz'], label='Without PCA', marker='o')
        plt.plot(self.metrics_with_pca['k'], self.metrics_with_pca['calinski_harabasz'], label='With PCA', marker='x')
        plt.title('Calinski-Harabasz Score Comparison')
        plt.xlabel('Number of Clusters k')
        plt.ylabel('Calinski-Harabasz Score')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
    def plot_elbow_method(self):
        ssd = []
        range_n_clusters = range(1, 11)
        for num_clusters in range_n_clusters:
            kmeans_temp = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans_temp.fit(self.features_scaled)
            ssd.append(kmeans_temp.inertia_)

        plt.figure(figsize=(8, 6))
        plt.plot(range_n_clusters, ssd, 'bo-', linewidth=2, markersize=5)
        plt.title('Elbow Method for Optimal Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Sum of Squared Distances')
        plt.grid(True)
        plt.show()
    


    def gdp_quarterly_feature_engineering(self):
        # Load GDP data
        gdp_data_path = "./dataset/combined_econ_data.csv"
        gdp_data = pd.read_csv(gdp_data_path, parse_dates=["DATE"], index_col="DATE")
        gdp_quarterly = gdp_data.resample("Q").last()

        # Initialize DataFrame for quarterly returns
        quarterly_returns_all_stocks = pd.DataFrame()

        # Process each stock file for quarterly returns
        for symbol, filename in self.stock_files.items():
            stock_data_path = f"./dataset/stock_data/{filename}"
            stock_data = pd.read_csv(
                stock_data_path, parse_dates=["Date"], index_col="Date"
            )
            daily_returns = stock_data["Close"].pct_change().dropna()
            quarterly_returns = daily_returns.resample("Q").mean()
            quarterly_returns_all_stocks[symbol] = quarterly_returns

        # Ensure there's no missing data across stocks
        quarterly_returns_all_stocks.dropna(inplace=True)

        # Prepare features for clustering: Mean of quarterly returns and GDP values
        self.features = pd.DataFrame(
            {
                "Mean_Quarterly_Returns": quarterly_returns_all_stocks.mean(axis=1),
                "GDP": gdp_quarterly["GDP"]
                .reindex(quarterly_returns_all_stocks.index)
                .ffill(),
            }
        ).dropna()

        # Scale features
        self.features_scaled = self.scaler.fit_transform(self.features)

    def gdp_quarterly_clustering_without_pca(self, n_clusters=4):
        # Clustering without PCA
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.features_scaled)

        # Evaluation metrics for without PCA version
        silhouette_avg = silhouette_score(self.features_scaled, clusters)
        davies_bouldin = davies_bouldin_score(self.features_scaled, clusters)
        calinski_harabasz = calinski_harabasz_score(self.features_scaled, clusters)

        # Print evaluation metrics for without PCA
        print(f"Without PCA - n_clusters={n_clusters}, Silhouette Score: {silhouette_avg:.2f}, Davies-Bouldin Index: {davies_bouldin:.2f}, Calinski-Harabasz Index: {calinski_harabasz:.2f}")

        # Plotting without PCA
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.features_scaled[:, 0], y=self.features_scaled[:, 1], hue=clusters, palette='viridis', legend='full', s=100)
        plt.title(f'Quarterly Stock Returns vs GDP Growth without PCA (n_clusters={n_clusters})')
        plt.xlabel('Mean Quarterly Returns')
        plt.ylabel('GDP Growth')
        plt.legend()
        plt.show()
    
    def gdp_quarterly_clustering_pca(self, n_clusters=4):

        pca = PCA(n_components=2)
        daily_returns_t = self.data.T
        pca.fit(daily_returns_t)
        daily_returns_pca = pca.transform(daily_returns_t)
        features_pca = pd.DataFrame(daily_returns_pca, index=self.data.columns, columns=['PC1', 'PC2'])
        features_pca_scaled = self.scaler.fit_transform(features_pca)


        # Clustering with PCA
        kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
        clusters_pca = kmeans_pca.fit_predict(features_pca_scaled)

        # Plotting with PCA
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=features_pca_scaled[:, 0], y=features_pca_scaled[:, 1], hue=clusters_pca, palette='viridis', legend='full',s=100)
        plt.title(f'Quarterly Stock Returns vs GDP Growth with PCA (n_clusters={n_clusters})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

        # Evaluation metrics for PCA version
        silhouette_avg_pca = silhouette_score(features_pca_scaled, clusters_pca)
        davies_bouldin_pca = davies_bouldin_score(features_pca_scaled, clusters_pca)
        calinski_harabasz_pca = calinski_harabasz_score(features_pca_scaled, clusters_pca)

        # Print evaluation metrics for with PCA
        print(f"With PCA - n_clusters={n_clusters}, Silhouette Score: {silhouette_avg_pca:.2f}, Davies-Bouldin Index: {davies_bouldin_pca:.2f}, Calinski-Harabasz Index: {calinski_harabasz_pca:.2f}")
   
    def economic_indicator_clustering_without_pca(self, indictors):
        # Load GDP and interest rate data
        combined_econ_data_path = './dataset/combined_econ_data.csv'
        econ_data = pd.read_csv(combined_econ_data_path, parse_dates=['DATE'], index_col='DATE')

        for index_name in indictors:
            # Prepare features for clustering
            features = pd.DataFrame({
                'Mean_Monthly_Returns': self.data.mean(axis=1),
                index_name: econ_data[index_name].reindex(self.data.index).ffill()
            }).dropna()

            # Scale features
            features_scaled = self.scaler.fit_transform(features)

            # Number of clusters
            n_clusters = 4

            # Clustering without PCA
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)

            # Evaluation metrics for without PCA version
            silhouette_avg = silhouette_score(features_scaled, clusters)
            davies_bouldin = davies_bouldin_score(features_scaled, clusters)
            calinski_harabasz = calinski_harabasz_score(features_scaled, clusters)

            # Print evaluation metrics for without PCA
            print(f"Without PCA - {index_name}: n_clusters={n_clusters}, Silhouette Score: {silhouette_avg:.2f}, Davies-Bouldin Index: {davies_bouldin:.2f}, Calinski-Harabasz Index: {calinski_harabasz:.2f}")

            # Plotting without PCA
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=features_scaled[:, 0], y=features_scaled[:, 1], hue=clusters, palette='viridis', legend='full', s=100)
            plt.title(f'Monthly Stock Returns vs {index_name} without PCA (n_clusters={n_clusters})')
            plt.xlabel('Mean Monthly Returns')
            plt.ylabel(index_name)
            plt.legend()
            plt.show()

    def economic_indicator_clustering_pca(self, indictors):
        # Load GDP and interest rate data
        combined_econ_data_path = './dataset/combined_econ_data.csv'
        econ_data = pd.read_csv(combined_econ_data_path, parse_dates=['DATE'], index_col='DATE')

        for index_name in indictors:
            # Prepare features for clustering
            features = pd.DataFrame({
                'Mean_Monthly_Returns': self.data.mean(axis=1),
                index_name: econ_data[index_name].reindex(self.data.index).ffill()
            }).dropna()

            # Scale features
            features_scaled = self.scaler.fit_transform(features)

            # Number of clusters
            n_clusters = 4
            
            # Proceed with PCA analysis and plotting as before
            # PCA
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features_scaled)

            # Clustering with PCA
            kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
            clusters_pca = kmeans_pca.fit_predict(features_pca)

            # Plotting with PCA
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=clusters_pca, palette='viridis', legend='full',s=100)
            plt.title(f'Monthly Stock Returns vs {index_name} with PCA (n_clusters={n_clusters})')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend()
            plt.show()

            # Evaluation metrics for PCA version
            silhouette_avg_pca = silhouette_score(features_pca, clusters_pca)
            davies_bouldin_pca = davies_bouldin_score(features_pca, clusters_pca)
            calinski_harabasz_pca = calinski_harabasz_score(features_pca, clusters_pca)

            # Print evaluation metrics for with PCA
            print(f"With PCA - {index_name}: n_clusters={n_clusters}, Silhouette Score: {silhouette_avg_pca:.2f}, Davies-Bouldin Index: {davies_bouldin_pca:.2f}, Calinski-Harabasz Index: {calinski_harabasz_pca:.2f}")



    def quarterly_feature_engineering(self, n_clusters=4):
        # Load GDP data
        gdp_data_path = "./dataset/stock_balance_sheet.csv"
        gdp_data = pd.read_csv(gdp_data_path, parse_dates=["DATE"], index_col="DATE")
        gdp_quarterly = gdp_data.resample("Q").last()

        # Initialize DataFrame for quarterly returns
        quarterly_returns_all_stocks = pd.DataFrame()

        # Process each stock file for quarterly returns
        for symbol, filename in self.stock_files.items():
            stock_data_path = f"./dataset/stock_data/{filename}"
            stock_data = pd.read_csv(
                stock_data_path, parse_dates=["Date"], index_col="Date"
            )
            daily_returns = stock_data["Close"].pct_change().dropna()
            quarterly_returns = daily_returns.resample("Q").mean()
            quarterly_returns_all_stocks[symbol] = quarterly_returns

        # Ensure there's no missing data across stocks
        quarterly_returns_all_stocks.dropna(inplace=True)

        financial_metrics = [
            "Revenue Growth (YoY)", "Shares Change", "Gross Margin", "Operating Margin",
            "Profit Margin", "Free Cash Flow Margin", "EBITDA Margin", "EBIT Margin",
            "Cash Growth", "Debt Growth"
        ]

        # Loop through each financial metric for feature engineering and clustering
        for metric in financial_metrics:
            # Prepare features
            self.features = pd.DataFrame({
                "Mean_Quarterly_Returns": quarterly_returns_all_stocks.mean(axis=1),
                metric: gdp_quarterly[metric]
                .reindex(quarterly_returns_all_stocks.index)
                .ffill(),
            }).dropna()

            # Scale features
            self.features_scaled = self.scaler.fit_transform(self.features)

            # Perform clustering without PCA
            self.quarterly_clustering_without_pca(n_clusters,metric=metric)
            self.quarterly_clustering_pca(n_clusters,metric=metric)



    def quarterly_clustering_without_pca(self, n_clusters=4, metric="metric"):
        # Clustering without PCA
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.features_scaled)

        # Evaluation metrics for without PCA version
        silhouette_avg = silhouette_score(self.features_scaled, clusters)
        davies_bouldin = davies_bouldin_score(self.features_scaled, clusters)
        calinski_harabasz = calinski_harabasz_score(self.features_scaled, clusters)

        # Print evaluation metrics for without PCA
        print(f"Without PCA - n_clusters={n_clusters}, Silhouette Score: {silhouette_avg:.2f}, Davies-Bouldin Index: {davies_bouldin:.2f}, Calinski-Harabasz Index: {calinski_harabasz:.2f}")

        # Plotting without PCA
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.features_scaled[:, 0], y=self.features_scaled[:, 1], hue=clusters, palette='viridis', legend='full', s=100)
        plt.title(f'Quarterly Stock Returns vs {metric} without PCA (n_clusters={n_clusters})')
        plt.xlabel('Mean Quarterly Returns')
        plt.ylabel(metric)
        plt.legend()
        plt.show()
    
    def quarterly_clustering_pca(self, n_clusters=4, metric="Metric"):
        # Initialize PCA with 2 components as we want to plot data in a 2D space
        pca = PCA(n_components=2)
        
        # Fit PCA on the scaled features
        pca.fit(self.features_scaled)
        
        # Transform the features using PCA
        features_pca = pca.transform(self.features_scaled)
        
        # Perform clustering on the PCA-transformed features
        kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
        clusters_pca = kmeans_pca.fit_predict(features_pca)
        
        # Plotting with PCA
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=clusters_pca, palette='viridis', legend='full', s=100)
        plt.title(f'PCA of Quarterly Stock Returns vs {metric} (n_clusters={n_clusters})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

        # Evaluation metrics for PCA version
        silhouette_avg_pca = silhouette_score(features_pca, clusters_pca)
        davies_bouldin_pca = davies_bouldin_score(features_pca, clusters_pca)
        calinski_harabasz_pca = calinski_harabasz_score(features_pca, clusters_pca)

        # Print evaluation metrics for with PCA
        print(f"With PCA - n_clusters={n_clusters}, Silhouette Score: {silhouette_avg_pca:.2f}, Davies-Bouldin Index: {davies_bouldin_pca:.2f}, Calinski-Harabasz Index: {calinski_harabasz_pca:.2f}")

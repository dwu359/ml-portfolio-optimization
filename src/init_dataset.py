import os
import pandas as pd


class InitDataset:
        
    @classmethod
    def create_econ_data(init=False):
        if not init:
            return
        # Define datasets paths
        datasets = [
            "dataset/econ_data/10yInterestrate_dataset.csv",
            "dataset/econ_data/GDP_dataset.csv",
            "dataset/econ_data/Median Consumer Price Index.csv",
            "dataset/econ_data/personal_saving_rate_dataset.csv",
            "dataset/econ_data/unemployed_dataset.csv",
        ]

        # Create an empty DataFrame to store merged data
        merged_df = pd.DataFrame(
            {"DATE": pd.date_range(start="2018-01-01", end="2023-12-31", freq="MS")}
        )
        merged_df.set_index("DATE", inplace=True)

        # Merge datasets based on 'DATE' column
        for path in datasets:
            temp_df = pd.read_csv(path)
            temp_df["DATE"] = pd.to_datetime(temp_df["DATE"])
            temp_df.set_index("DATE", inplace=True)

            # Merge on 'DATE' column
            merged_df = pd.merge(
                merged_df, temp_df, left_index=True, right_index=True, how="left"
            )

        # Fill missing values with values from the previous month
        merged_df = merged_df.reset_index()
        merged_df.ffill(inplace=True)

        # Copy monthly value to convert into daily format
        pivot_df = merged_df.pivot(index="DATE", columns="GDP")

        start_date = pivot_df.index.min() - pd.DateOffset(day=1)
        end_date = pivot_df.index.max() + pd.DateOffset(day=31)
        dates = pd.date_range(start_date, end_date, freq="D")
        dates.name = "DATE"
        merged_econind_df = (
            pivot_df.reindex(dates, method="ffill").stack("GDP").reset_index()
        )
        merged_econind_df.set_index("DATE", inplace=True)

        # Save the merged DataFrame to a CSV file
        merged_econind_df.to_csv(
            "./dataset/econ_data/combined_econ_data.csv", index=True
        )

    @classmethod
    def create_merged_stock_data(init=False):
        if not init:
            return
        datasets = [
            "dataset/stock_data/AAPL_data.csv",
            "dataset/stock_data/AMD_data.csv",
            "dataset/stock_data/AMZN_data.csv",
            "dataset/stock_data/F_data.csv",
            "dataset/stock_data/GOOG_data.csv",
            "dataset/stock_data/INTC_data.csv",
            "dataset/stock_data/JPM_data.csv",
            "dataset/stock_data/MS_data.csv",
            "dataset/stock_data/MSFT_data.csv",
            "dataset/stock_data/NVDA_data.csv",
            "dataset/stock_data/TSLA_data.csv",
            "dataset/stock_data/VOO_data.csv",
        ]
        dfs = []
        for dataset in datasets:
            temp_df = pd.read_csv(dataset)
            temp_df["Date"] = pd.to_datetime(temp_df["Date"])
            temp_df.set_index("Date", inplace=True)
            dfs.append(temp_df)
        # Merge on 'DATE' column

        # Not sure how useful this DF will be, but it exists now
        merged_stocktick_df = pd.concat(dfs)

        merged_stocktick_df.to_csv("./dataset/combined_stock_data.csv", index=True)

    @classmethod
    def create_adj_closed_price(init=False):
        if not init:
            return
        directory = 'dataset/stock_data'

        dfs = []
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(directory, filename))
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                # Check for duplicate index values
                if df.index.duplicated().any():
                    df = df[~df.index.duplicated()]
                    print(f"Warning: Duplicate index values in file {filename}. Duplicates have been dropped.")
                
                adj_close_column = df['Adj Close']
                ticker = df['ticker'].unique()[0]
                adj_close_column.name = ticker
                dfs.append(adj_close_column)


        combined_df = pd.concat(dfs, axis=1)
        combined_df.to_csv('./dataset/stock_data/combined_stock_adj_closed.csv', index=True)
    
    def get_test_train_data():
        dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')
        price_data = pd.read_csv(r'dataset/stock_data/combined_stock_adj_closed.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
        train_data = price_data[price_data.index < '2023-01-01']
        test_data = price_data[price_data.index >= '2023-01-01']
        index_data = pd.read_csv(r'dataset/stock_data/SPY_data.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)['Adj Close']
        return test_data, train_data, index_data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


def prepare_data(data, seq_len=100):
    x = []
    y = []
    for i in range(len(data) - seq_len - 1):
        x.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(x), np.array(y)


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    @staticmethod
    def build_model(
        input_size=1, hidden_size=1, num_layers=1, num_classes=1, learning_rate=0.001
    ):
        model = MyLSTM(input_size, hidden_size, num_layers, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        return model, optimizer, criterion

    @staticmethod
    def train_model(model, optimizer, criterion, x_train, y_train, num_epochs=1000):
        loss_track = []

        for epoch in range(num_epochs + 1):
            outputs = model.forward(x_train.to(device))
            optimizer.zero_grad()
            loss = criterion(outputs, y_train.to(device))
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                loss_track.append(loss.item())
                print(f"Epoch {epoch}/{num_epochs}, Loss={loss.item()}")
        return model, loss_track

    @staticmethod
    def fit(ticker, train_data, seq_len, num_epochs):
        scalar = MinMaxScaler()
        train_data = scalar.fit_transform(np.array(train_data[ticker]).reshape(-1, 1))
        x_train, y_train = prepare_data(train_data, seq_len)

        X_train_tensors = Variable(torch.Tensor(x_train))
        y_train_tensors = Variable(torch.Tensor(y_train))
        model, optimizer, criterion = MyLSTM.build_model()
        model, loss_track = MyLSTM.train_model(
            model, optimizer, criterion, X_train_tensors, y_train_tensors, num_epochs
        )
        return model, loss_track, scalar

    @staticmethod
    def fit_all_tickers(train_data, seq_len, num_epochs):
        models = {}
        all_loss_tracks = {}
        scalars = {}
        for ticker in train_data.columns:
            print("-" * 50)
            print(f"Fitting model for ticker: {ticker}")
            print("-" * 50)
            model, loss_track, scalar = MyLSTM.fit(
                ticker, train_data, seq_len, num_epochs
            )
            models[ticker] = model
            all_loss_tracks[ticker] = loss_track
            scalars[ticker] = scalar
        return models, all_loss_tracks, scalars

    @staticmethod
    def plot_predicted_vs_actual(y_test, y_pred, ticker):
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.cpu().numpy(), label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.title(f"Predicted vs Actual for {ticker}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_loss_track(loss_track, ticker):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_track)
        plt.title(f"Loss track for {ticker}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

    @staticmethod
    def test(models, val_data, test_data, all_loss_tracks, scalars, seq_length=100):
        criterion = nn.MSELoss()

        # DataFrames to store predicted values and losses
        predicted_df = pd.DataFrame(index=val_data.index)
        print(predicted_df.shape)
        test_data = test_data.head(len(val_data))
        loss_df = pd.DataFrame(columns=models.keys())
        for ticker in models:
            model = models[ticker]
            scalar = scalars[ticker]
            data = val_data[ticker]
            y_test = torch.Tensor(test_data[ticker][:seq_length])
            print(f"Evaluating model for ticker: {ticker}")
            with torch.no_grad():
                # Assuming x_test and y_test are available from test_data
                test_inputs = list(
                    scalar.transform(data[-seq_length:].values.reshape(-1, 1)).reshape(
                        -1
                    )
                )
                for _ in range(seq_length):
                    seq = (
                        torch.tensor(test_inputs[-seq_length:])
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(2)
                    ).to(device)
                    test_inputs.append(model(seq).item())
                y_pred = np.array(test_inputs[seq_length:])
                print(len(y_pred))
                y_pred_original_scale = scalar.inverse_transform(
                    y_pred.reshape(-1, 1)
                ).reshape(-1)

                # print(y_pred_original_scale.shape)
                # print(y_test.shape)
                loss = criterion(torch.Tensor(y_pred_original_scale), y_test)
                predicted_df[ticker] = y_pred_original_scale

                # Plot predicted vs actual
                MyLSTM.plot_predicted_vs_actual(y_test, y_pred_original_scale, ticker)

                # Plot loss track
                MyLSTM.plot_loss_track(all_loss_tracks[ticker], ticker)

                loss_df[ticker] = all_loss_tracks[ticker]

        return predicted_df, loss_df

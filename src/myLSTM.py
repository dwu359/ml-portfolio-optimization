import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_data(data, seq_len=100):
    x = []
    y = []
    for i in range(len(data)-seq_len-1):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return torch.tensor(x).float(), torch.tensor(y).float()

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
    def build_model(input_size=1, hidden_size=1, num_layers=1, num_classes=1, learning_rate=0.001):
        model =  MyLSTM(input_size, hidden_size, num_layers, num_classes)
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
                print(f'Epoch {epoch}/{num_epochs}, Loss={loss.item()}')
        return model, loss_track


    @staticmethod
    def fit(ticker, train_data, test_data, seq_len, num_epochs):
        scalar = MinMaxScaler()
        train_data = scalar.fit_transform(np.array(train_data[ticker]).reshape(-1, 1))
        test_data = scalar.transform(np.array(test_data[ticker]).reshape(-1, 1))

        x_train, y_train = prepare_data(train_data, seq_len)
        x_test, y_test = prepare_data(test_data, seq_len)
        model, optimizer, criterion = MyLSTM.build_model()
        model, loss_track = MyLSTM.train_model(model, optimizer, criterion, x_train, y_train, num_epochs)
        
        y_pred = model(x_test.to(device))
        y_pred = scalar.inverse_transform(y_pred.detach().cpu().numpy())
        return model, loss_track, y_pred

    
    @staticmethod
    def fit_all_tickers(train_data, test_data, seq_len, num_epochs):
        models = {}
        all_loss_tracks = {}
        all_y_preds = {}
        for ticker in train_data.columns[1:]:
            print("-"*50)
            print(f"Fitting model for ticker: {ticker}")
            print("-"*50)
            model, loss_track, y_pred = MyLSTM.fit(ticker, train_data, test_data, seq_len, num_epochs)
            models[ticker] = model
            all_loss_tracks[ticker] = loss_track
            all_y_preds[ticker] = y_pred
        return models, all_loss_tracks, all_y_preds
    
    @staticmethod
    def plot_predicted_vs_actual(y_test, y_pred, ticker):
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.cpu().numpy(), label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(f'Predicted vs Actual for {ticker}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_loss_track(loss_track, ticker):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_track)
        plt.title(f'Loss track for {ticker}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    @staticmethod
    def test(models, test_data, all_loss_tracks, all_y_preds):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.MSELoss()
        
        # DataFrames to store predicted values and losses
        predicted_df = pd.DataFrame(index=test_data.index[:-1])
        loss_df = pd.DataFrame(columns=models.keys())
        
        for ticker, model in models.items():
            print(f"Evaluating model for ticker: {ticker}")
            with torch.no_grad():
                # Assuming x_test and y_test are available from test_data
                x_test = torch.Tensor(test_data[ticker].values[:-1]).unsqueeze(0).unsqueeze(2)
                x_test = x_test.permute(1, 0, 2).to(device)
                y_test = torch.Tensor(test_data[ticker].values[1:]).to(device)
                
                # Forward pass
                y_pred = model(x_test)
                y_pred = y_pred.squeeze().detach().cpu().numpy()
                
                # Calculate loss
                loss = criterion(torch.Tensor(y_pred), y_test)
                print(f"Loss for {ticker}: {loss.item()}")
                
                # Store predicted values and loss
                predicted_df[ticker] = y_pred
                loss_df[ticker] = all_loss_tracks[ticker]
                
                # Plot predicted vs actual
                MyLSTM.plot_predicted_vs_actual(y_test, y_pred, ticker)
                
                # Plot loss track
                # MyLSTM.plot_loss_track(all_loss_tracks[ticker], ticker)
                
        return predicted_df, loss_df




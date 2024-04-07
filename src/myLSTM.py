import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.autograd as autograd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_data(data, seq_len=100):
    x = []
    y = []
    for i in range(len(data)-seq_len-1):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return torch.tensor(x).float(), torch.tensor(y).float()

def split_train_test(data):
    train = data[data['Date'] < '2023']
    test = data[data['Date'] >= '2023']
    return train, test

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
    
    def build_model(input_size = 1, hidden_size = 1, num_layers = 1, num_classes = 1, learning_rate = 0.001):
        model =  MyLSTM(input_size, hidden_size, num_layers, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        return model, optimizer, criterion
    
    def train_model(ticker, x_train, y_train, num_epochs = 1000):
        model,optimizer,criterion = MyLSTM.build_model()
        model_loss_df = pd.DataFrame(columns=['ticker'])
        loss_track = []

        for epoch in range(num_epochs+1):
            outputs = model.forward(x_train.to(device))
            optimizer.zero_grad()
            loss = criterion(outputs, y_train.to(device))
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                loss_track.append(loss.item())
                print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item()}')
        return model, loss_track

    def fit(ticker, data, seq_len, num_epochs):
        predictions_df = pd.DataFrame(columns=['ticker'])
        scalar = MinMaxScaler()
        data = scalar.fit_transform(np.array(data[ticker]).reshape(-1, 1))

        train , test = split_train_test(data)
        x_train, y_train = prepare_data(train, seq_len)
        x_test, y_test = prepare_data(test, seq_len)
        model, loss_track = MyLSTM.train_model(ticker, x_train, y_train, num_epochs)
        
        y_pred = model(x_test.to(device))
        y_pred = scalar.inverse_transform(y_pred.detach().cpu().numpy())
        return model, loss_track, y_pred
    
    def fit_all_tickers(data, seq_len, num_epochs):
        models = {}
        for ticker in data.columns[1:]:
            model, loss_track, y_pred = MyLSTM.fit(ticker, data, seq_len, num_epochs)
            models[ticker] = model
        return models

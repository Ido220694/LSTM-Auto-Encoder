import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as op
import torch.nn as nn
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


class AE(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AE, self).__init__()
        self.encoder_LSTM = nn.LSTM(1, hidden_layer_size, batch_first=True)
        self.decoder_LSTM = nn.LSTM(hidden_layer_size, hidden_layer_size, batch_first=True)
        self.hidden_layer_size = hidden_layer_size
        self.func = nn.Linear(hidden_layer_size,1)
        self.forecast = nn.Linear(hidden_layer_size, 1)

    def forward(self, x_t):
        x, (z,y) = self.encoder_LSTM(x_t)
        z = z.reshape(-1, 1, self.hidden_layer_size).repeat(1, x_t.size(1) , 1)
        h_temp , s =self.decoder_LSTM(z)
        return self.func(h_temp), self.forecast(h_temp)

class LSTM_AE_SP500(Dataset):
    def __init__(self, t):
        self.our_list = list(t.index)
        x = minmax_scale(t, axis=1)
        x = x - x.mean(axis=1, keepdims=True) + 0.5
        self.data = []
        self.tags = []
        for i in x:
            self.data.extend([torch.tensor(i[:-1]).unsqueeze(1)])
            self.tags.extend([torch.tensor(i[1:]).unsqueeze(1)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.our_list[item], self.tags[item] ,self.data[item]



def training(epochs, optim, model, clip, data, values):
        curr_loss = []
        forc_loss =[]
        val_curr_loss = []
        val_forc_loss = []
        adder = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_func = nn.MSELoss()

        for e in range(1, epochs):
           current_loss = 0
           forecast_loss =0
           for els, datum, tags in data:
                # zero the parameter gradients
                optim.zero_grad()
                tags = tags.to(adder)
                datum = datum.to(adder)
                outputs, forecast = model(datum)
                reg_loss = loss_func(outputs, datum)
                new_loss = loss_func(forecast, tags)
                loss = reg_loss + new_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optim.step()
                current_loss = current_loss +reg_loss.item()
                forecast_loss = forecast_loss + new_loss.item()

           new_item = current_loss/len(data)
           new_item_3 = forecast_loss / len(data)
           curr_loss.append(new_item)
           forc_loss.append(new_item_3)

           val_current_loss = 0
           val_forecast_loss = 0
           with torch.no_grad():
                for _, datum, tags in values:
                    tags = tags.to(adder)
                    datum = datum.to(adder)
                    outputs, forecast = model(datum)
                    reg_loss = loss_func(outputs, datum)
                    new_loss = loss_func(forecast, tags)
                    val_current_loss = val_current_loss + reg_loss.item()
                    val_forecast_loss = val_forecast_loss + new_loss.item()

                new_item_2 = val_current_loss / len(data)
                new_item_4 = val_forecast_loss / len(data)
                val_curr_loss.append(new_item_2)
                val_forc_loss.append(new_item_4)
           print("~~~~~~~~~~~~~~~~~")
           print(f" Epoch {e}\n train loss: {new_item:.3f}\n val loss: {new_item_2:.3f}\n")
           print(f"train accuracy: {new_item_3:.3f}\n validation acc: {new_item_4:.3f}\n")

        return curr_loss, forc_loss, val_curr_loss, val_forc_loss

parser= argparse.ArgumentParser()
parser.add_argument("--epochs", type = int, default = 1300)
parser.add_argument("--optim", choices =["sgd", "adam"], default="adam")
parser.add_argument("--clip", type=float, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=0.001)

arguments = parser.parse_args()

print(f"parameters: epochs:{arguments.epochs} optimizer:{arguments.optim} "
      f"learning rate:{arguments.lr} clipping:{arguments.clip} batch_size:{arguments.batch_size}")

model = AE(arguments.hidden_size)
adder = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(adder)
sp500 = pd.read_csv("./data/SP 500 Stock Prices 2014-2017.csv", parse_dates=["date"], dtype={"close":np.float32})

prices = sp500.pivot(index="symbol", columns="date", values="close").dropna()

train_set, test_set = train_test_split(prices, test_size=0.4)
val_set, test_set = train_test_split(test_set, test_size=0.5)

new = sp500.set_index('symbol')
new2 = new.loc[['AMZN','GOOGL', 'APC' ,'TXN', 'PPL', 'NOV'], :]

prices2 = new2.pivot_table(index="symbol", columns="date", values="close").dropna()

data = DataLoader(LSTM_AE_SP500(train_set), batch_size=arguments.batch_size, shuffle=True)
values = DataLoader(LSTM_AE_SP500(val_set), batch_size=arguments.batch_size, shuffle=False)
check = DataLoader(LSTM_AE_SP500(test_set), batch_size=arguments.batch_size, shuffle=False)
check_2 = DataLoader(LSTM_AE_SP500(prices2), batch_size=arguments.batch_size, shuffle=False)

if arguments.optim == "sgd":
    optimizer = op.SGD(model.parameters(), lr=arguments.lr)
else:
    optimizer = op.Adam(model.parameters(), lr=arguments.lr)

curr_loss, forc_loss, val_curr_loss, val_forc_loss = training(arguments.epochs, optimizer, model, arguments.clip, data, values)

#Date of the Training
plt.title("Training Loss per epoch")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.plot(val_curr_loss, label="Valid")
plt.plot(curr_loss, label="Train")
plt.legend()
plt.show()

plt.title("Forecast per epoch")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.plot(val_forc_loss, label="Valid")
plt.plot(forc_loss, label="Train")
plt.legend()
plt.show()

#Output and Forecast

for i in range(5):

    with torch.no_grad():
        for els, datum, tags in check_2:
            datum = datum.to(adder)
            outputs, forecast = model(datum)
            temp = outputs.squeeze(0)[i]
            temp2 = forecast.squeeze(0)[i]
            g1 = temp.detach().numpy()
            g2 = temp2.detach().numpy()
            print(g1)
            print(g2)
    dat = datum.squeeze(0)[i]
    f = dat.detach().numpy()
    plt.plot(f, label="input")
    plt.plot(g1, label="output")
    plt.legend()
    plt.show()

    plt.plot(f, label="input")
    plt.plot(g2, label="forecast")
    plt.legend()
    plt.show()






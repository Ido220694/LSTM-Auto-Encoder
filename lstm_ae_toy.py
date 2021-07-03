import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as op
import torch.nn as nn
import pandas as pd
from matplotlib import pyplot as plt
import math
import random


class AE(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AE, self).__init__()
        self.encoder_LSTM = nn.LSTM(1, hidden_layer_size, batch_first=True)
        self.decoder_LSTM = nn.LSTM(hidden_layer_size, hidden_layer_size, batch_first=True)
        self.hidden_layer_size = hidden_layer_size
        self.func = nn.Linear(hidden_layer_size,1)

    def forward(self, x_t):
        x, (z,y) = self.encoder_LSTM(x_t)
        z = z.reshape(-1, 1, self.hidden_layer_size).repeat(1, x_t.size(1) , 1)
        h_temp , s =self.decoder_LSTM(z)
        return self.func(h_temp)


class LSTM_AE_TOY(Dataset):
    def __init__(self, t):
        self.data = np.expand_dims(t.values, 2).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def grid_search(values, data):
        print("grif")
        hidden_state_size = [ 32, 64, 128, 256]
        learning_rates =[ 0.0001, 0.001, 0.01 , 0.1 ]
        gradient_clipping = [0.01, 0.1, 0.5, 1]

        min_loss = math.inf
        min_losses = []
        min_train =[]
        chosen_lr =0
        chosen_clipping = 0
        chosen_hidden =0

        for clip in gradient_clipping:
            for learn in learning_rates:
                for hidden in hidden_state_size:
                    model = AE(hidden)
                    optim = op.Adam(model.parameters(), lr=learn)
                    tr_los, v_los = training(150, optim, model, hidden, clip, data, values)
                    if min_loss > v_los[len(v_los)-1]:
                        chosen_clipping = clip
                        chosen_lr = learn
                        chosen_hidden = hidden
                        min_loss = v_los[len(v_los)-1]
                        min_losses = v_los
                        min_train = tr_los

        return [chosen_clipping, chosen_lr, chosen_hidden, min_loss, min_losses, min_train]


def training(epchos,optim, model, hidden, clip, data, values):
        vals = []
        training =[]
        adder = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_func = nn.MSELoss()

        for e in range(1, epchos):
           total_loss =0
           total_loss_2 = 0
           for datum in data:
               # zero the parameter gradients
                optim.zero_grad()
                datum = datum.to(adder)
                outputs = model(datum)
                loss= loss_func(outputs, datum)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optim.step()
                total_loss = total_loss + loss.item()
           with torch.no_grad():
                for val in values:
                    val = val.to(adder)
                    outputs = model(val)
                    loss = loss_func(outputs, val)
                    total_loss_2 = total_loss_2 + loss.item()

           new_item = total_loss/len(data)
           training.append(new_item)
           new_item2 = total_loss_2 / len(values)
           vals.append(new_item2)
           print("~~~~~~~~~~~~~~~~~")
           print(f" Epoch {e}\n train loss: {new_item:.3f}\n val loss: {new_item2:.3f}\n ")

        return training, vals


parser= argparse.ArgumentParser()
parser.add_argument("--epchos", type = int, default = 150)
parser.add_argument("--grid_search", action="store_const", const= True)
parser.add_argument("--optim", choices =["sgd", "adam"], default="adam")
parser.add_argument("--clip", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=64 )
parser.add_argument("--hidden_size", type=int, default= 128)
parser.add_argument("--lr", type=float, default=0.01)

arguments = parser.parse_args()
data = DataLoader(LSTM_AE_TOY(pd.read_csv("./data/toy_train.csv")), batch_size=arguments.batch_size, shuffle=True)
values = DataLoader(LSTM_AE_TOY(pd.read_csv("./data/toy_val.csv")), batch_size=arguments.batch_size, shuffle=False)


if arguments.grid_search:
    print("here")
    grid_search_res =grid_search(values, data)
    print(f"grid_search outputs:: chosen clipping:{grid_search_res[1]} learning rate:{grid_search_res[1]}"
           f"chosen hidden:{grid_search_res[2]} min loss:{grid_search_res[3]}")
    plt.title("Loss per epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(grid_search_res[4], label="Valid")
    plt.plot(grid_search_res[5], label="Train")
    plt.legend()
    plt.show()

else:

    print("stop")
    model = AE(arguments.hidden_size)
    adder = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(adder)

    print(f"parameters: epcohs:{arguments.hidden_size} optimizer:{arguments.optim} learning rate:{arguments.lr} clipping:{arguments.clip} batch_size:{arguments.batch_size}")
    if arguments.optim == "sgd":
        optimizer = op.SGD(model.parameters(), lr=arguments.lr)
    else:
        optimizer = op.Adam(model.parameters(), lr=arguments.lr)

    training , vals= training(arguments.epchos, optimizer,model, arguments.hidden_size, arguments.clip, data, values)
    plt.title("Training Loss per epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(vals, label="Valid")
    plt.plot(training, label="Train")
    plt.legend()
    plt.show()

    for i in range (4):

        datum = random.choice(next(iter(values)))

        g = []
        for i in range (50):
           g.append(datum.numpy()[i])


        plt.plot(g, label="input")
        plt.title("input vs output")

        output = model(datum.unsqueeze(0))
        t= output.detach().numpy().squeeze(0)
        plt.plot(t, label="output")
        plt.legend()
        plt.show()



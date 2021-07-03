import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as op
import torch.nn as nn
import pandas as pd
from matplotlib import pyplot as plt
import math
from torchvision import datasets, transforms
import random



class AE(nn.Module):
    def __init__(self, hidden_layer_size, n_inputs):
        super(AE, self).__init__()
        self.encoder_LSTM = nn.LSTM(n_inputs, hidden_layer_size, batch_first=True)
        self.decoder_LSTM = nn.LSTM(hidden_layer_size, hidden_layer_size, batch_first=True)
        self.hidden_layer_size = hidden_layer_size
        self.n_inputs = n_inputs
        self.func = nn.Linear(hidden_layer_size, n_inputs)
        self.indicator = nn.Linear(hidden_layer_size, 10)


    def forward(self, x_t):
        x, (z, y) = self.encoder_LSTM(x_t)
        z = z.view(-1, 1, self.hidden_layer_size).repeat(1, x_t.size(1) , 1)
        h_temp , s = self.decoder_LSTM(z)
        return self.func(h_temp), self.indicator(h_temp)

class LSTM_AE_MNIST(Dataset):
    def __init__(self, k, *, flatten=False):
        self.data = []
        self.tags = []
        if flatten:
            for (i,j) in k:
                self.data.extend([i.reshape(-1,1)])
                self.tags.extend([j])
        else:
            for (i,j) in k:
                self.data.extend([i.squeeze(0)])
                self.tags.extend([j])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return  self.data[item], self.tags[item]


def training(epochs, optim,  model,clip, data, values):

        vals = []
        training =[]
        training_success = []
        validation_success = []
        adder = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_func = nn.MSELoss()
        cross_entry_point = nn.CrossEntropyLoss()
        # print("he")
        for e in range(1, epochs):
           total_acc = 0
           total_loss =0
           total_all = 0

           for datum, tags in data:
               # print("ha")
               optim.zero_grad()
               tags = tags.to(adder)
               datum = datum.to(adder)
               outputs, new_input = model(datum)
               loss = loss_func(outputs, datum)
               new_loss = cross_entry_point(new_input.reshape(len(datum), -1), tags)
               loss = loss + new_loss
               loss.backward()
               nn.utils.clip_grad_norm_(model.parameters(), clip)
               optim.step()
               total_loss = total_loss + loss.item()
               with torch.no_grad():
                   _, new_acc = torch.max(new_input.reshape(len(datum),-1) , 1)
                   total_acc = total_acc + (new_acc == tags).sum().item()
                   total_all = total_all + len(datum)
               # print(total_acc)
               # print(total_all)

           with torch.no_grad():
               total_loss_2 = 0
               total_acc_2 = 0
               total_all_2 = 0
               for datum, tags in values:
                    tags = tags.to(adder)
                    datum = datum.to(adder)
                    outputs, new_input = model(datum)
                    loss = loss_func(outputs, datum)
                    new_loss = cross_entry_point(new_input.view(len(datum), -1), tags)
                    loss = loss + new_loss
                    total_loss_2 = total_loss + loss.item()
                    _, new_acc = torch.max(new_input.view(len(datum), -1), 1)
                    total_acc_2 = total_acc_2 + (new_acc == tags).sum().item()
                    total_all_2 = total_all_2 + len(datum)

           new_item = total_loss/len(data)
           new_item_3 = total_acc / total_all
           training.append(new_item)
           training_success.append(new_item_3)

           new_item2 = total_loss_2/len(values)
           new_item_4 = total_acc_2 / total_all_2
           vals.append(new_item2)
           validation_success.append((new_item_4))

           print("~~~~~~~~~~~~~~~~~")
           print(f" Epoch {e}\n train loss: {new_item:.3f}\n val loss: {new_item2:.3f}\n")
           print(f"train accuracy: {new_item_3:.3f}\n validation acc: {new_item_4:.3f}\n")

        return training, vals, training_success, validation_success



parser= argparse.ArgumentParser()
parser.add_argument("--epochs", type = int, default = 30)
parser.add_argument("--input_size", type = int, choices=[1,28], default = 1)
parser.add_argument("--optim", choices =["sgd", "adam"], default="adam")
parser.add_argument("--clip", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=0.001)

arguments = parser.parse_args()

tran = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST("./data", train = True, download = True, transform =tran)
mnist_test = list(datasets.MNIST("./data", train = False, download = True, transform =tran))

print(f"parameters: epochs:{arguments.epochs} input_size: {arguments.input_size}optimizer:{arguments.optim} "
      f"learning rate:{arguments.lr} clipping:{arguments.clip} batch_size:{arguments.batch_size} hidden_size:{arguments.hidden_size}")
if arguments.input_size == 1:
    data = DataLoader(LSTM_AE_MNIST(mnist_train, flatten=True), batch_size=arguments.batch_size, shuffle=True)
    values = DataLoader(LSTM_AE_MNIST(mnist_test[:5000], flatten=True), batch_size=arguments.batch_size,
                        shuffle=False)
    check = DataLoader(LSTM_AE_MNIST(mnist_test[5000:], flatten=True), batch_size=arguments.batch_size,
                        shuffle=False)

else:
    data = DataLoader(LSTM_AE_MNIST(mnist_train), batch_size=arguments.batch_size, shuffle=True)
    values = DataLoader(LSTM_AE_MNIST(mnist_test[:5000]), batch_size=arguments.batch_size,
                        shuffle=False)
    check = DataLoader(LSTM_AE_MNIST(mnist_test[5000:]), batch_size=arguments.batch_size,
                        shuffle=False)

adder = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AE(arguments.hidden_size, arguments.input_size).to(adder)

if arguments.optim == "sgd":
    optim = op.SGD(model.parameters(), lr=arguments.lr)
else:
    optim = op.Adam(model.parameters(), lr=arguments.lr)

training , vals,training_success, validation_success = training(arguments.epochs, optim, model, arguments.clip, data, values)

check_2 =0
check_all =0

with torch.no_grad():
    model = AE(arguments.hidden_size, arguments.input_size)
    adder = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(adder)
    for datum, tags in check:
        datum = datum.to(adder)
        tags = tags.to(adder)
        outputs, new_input = model(datum)
        _, new_acc = torch.max(new_input.reshape(len(datum), -1), 1)
        check_2 = check_2 + (new_acc == tags).sum().item()
        check_all = check_all + len(datum)
    acc = check_2 / check_all
    print(f"acc : {acc}.3f")

plt.title("Success per epoch")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.plot(validation_success, label="Valid")
plt.plot(training_success, label="Train")
plt.legend()
plt.show()

plt.title("Training Loss per epoch")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.plot(vals, label="Valid")
plt.plot(training, label="Train")
plt.legend()
plt.show()



datum = random.choice(next(iter(check))[0])
fig, res = plt.subplots(1,2, constrained_layout=True)
res[0].imshow(datum.numpy(), cmap="binary")
res[0].set_title("input")
res[0].axis("off")

with torch.no_grad():
    out = model(datum.unsqueeze(0))[0][0]
    res[1].imshow(out.cpu().numpy(), cmap="binary")
res[1].axis("off")
res[1].set_title("output")
plt.show()


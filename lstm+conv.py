import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

seq_len = 200
epochs = 1
cycles = 1
load = False

device = torch.device('cuda')
df = pd.read_feather('merged.feather').values[:, 1:].astype(float) * 10000
train = df[:-1000]
test = df[-1000 - seq_len:]


def get_stds(epochs):
    x_std = []
    y_std = []
    start = time.time()
    for i in range(epochs):
        for batch in train_loader:
            x, y, _ = batch
            x_std.append(x.std(0))
            y_std.append(y.std())
        print(i, time.time()-start)
    xtd = [i[None] for i in x_std]
    ytd = [i[None] for i in y_std]
    pickle.dump((torch.cat(xtd).mean(0), torch.cat(ytd).mean(0)), open('std.pkl', 'wb'))


class forex_data(Dataset):
    def __init__(self, source):
        super().__init__()
        self.data = source

    def __len__(self):
        return len(self.data) - seq_len - 1

    def __getitem__(self, item):
        diff = self.data[item + seq_len + 1] - self.data[item + seq_len]
        x_diff = self.data[item + 1:item + seq_len + 1] - self.data[item]
        return x_diff, np.append(diff[-2], -diff[-2]), diff[-2]


class forex(nn.Module):
    def __init__(self):
        super().__init__()
        self.upscale = nn.Sequential(nn.Linear(65, 130), nn.ELU())
        self.lstm = nn.LSTM(130, 130, batch_first=True, num_layers=2, dropout=0.1)
        self.conv = nn.Sequential(nn.Conv1d(200, 200, 5, 5), nn.MaxPool1d(13))
        self.conv1 = nn.Sequential(nn.Conv1d(120, 150, 5, 5), nn.MaxPool1d(13))
        self.conv2 = nn.Sequential(nn.Conv1d(50, 80, 5, 5), nn.MaxPool1d(13))
        self.conv3 = nn.Sequential(nn.Conv1d(10, 40, 5, 5), nn.MaxPool1d(13))
        self.predict = nn.Sequential(#nn.Linear(500, 1000), nn.ELU(), nn.Linear(1000, 500), nn.ELU(),
                                     nn.Linear(600, 500), nn.ELU(), nn.Linear(500, 250), nn.ELU(), nn.Linear(250, 50),
                                     nn.ELU(), nn.Linear(50, 2))

    def init_hidden(self):
        init = torch.zeros(2, self.batch_size, 130).to(device).float()
        self.hidden = (init, init)

    def forward(self, input):
        self.batch_size = input.shape[0]
        self.init_hidden()
        lstm, self.hidden = self.lstm(self.upscale(input[:,-10:, :]), self.hidden)
        lstm = lstm[:, -1, :].view(-1, 130)
        conv = self.conv(input).view(self.batch_size, -1)
        conv1 = self.conv1(input[:, -120:, :]).view(self.batch_size, -1)
        conv2 = self.conv2(input[:, -50:, :]).view(self.batch_size, -1)
        conv3 = self.conv3(input[:, -10:, :]).view(self.batch_size, -1)
        cat = torch.cat((lstm, conv, conv1, conv2, conv3), dim=1)
        return self.predict(cat)


trainset = forex_data(train)
testset = forex_data(test)
train_loader = DataLoader(trainset, batch_size=5000, shuffle=True, num_workers=2)
test_loader = DataLoader(testset, batch_size=100000)

net = forex().to(device)
if load:
    net.load_state_dict(torch.load('lstm_conv1.pt'))
    print('Loaded net')
criterion = nn.MSELoss()

start = time.time()
train_loss = []
test_loss = []
x_std, y_std = pickle.load(open('std.pkl', 'rb'))
x_std, y_std = x_std.to(device).float(), y_std.to(device).float()
for cycle in range(cycles):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    for epoch in range(epochs):
        batch_loss = []
        for i, batch in enumerate(train_loader):
            x, y, _ = batch
            x, y = x.to(device).float(), y.to(device).float()
            x = x / x_std
            pred = net(x)
            loss = criterion(pred, y / y_std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        train_loss.append(np.mean(batch_loss))
        scheduler.step()
        for test_batch in test_loader:
            x_t, y_t, pip = test_batch
            x_t, y_t = x_t.to(device).float(), y_t.to(device).float()
            x_t = x_t / x_std
            pred_t = net(x_t)
            loss_t = criterion(pred_t, y_t / y_std)
        test_loss.append(loss_t.item())
        print(cycle + 1, epoch + 1, time.time() - start, np.mean(batch_loss), loss_t.item())

plt.plot(train_loss, color='blue')
plt.plot(test_loss, color='orange')
plt.show()

pred_t_ = pred_t.detach().cpu().numpy()
ap = np.where(np.argmax(pred_t_, 1) < 0.5, 1, -1) * pip.cpu().numpy()
plt.plot([np.sum(ap[:i]) for i in range(len(ap))])
plt.show()
y_t = y_t.cpu().detach().numpy()
pip = pip.cpu().numpy()
results = pd.DataFrame(np.concatenate((pred_t_, y_t / y_std, pip[None].T), axis=1))
torch.save(net.state_dict(), 'lstm_conv1.pt')
print('saved')

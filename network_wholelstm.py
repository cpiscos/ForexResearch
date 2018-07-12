import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import time

save = True
load = True
network_file = 'saves/network_lstm1.pt'
graph_file = 'saves/graph_lstm1.pkl'
batch_size = 1
epochs = 1
cycles = 1
seq_len = 80000

device = torch.device('cuda')
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
df = pd.read_feather('merged.feather')
pred_hours = 1
# -8302 - seq_len - pred_hours
# -9302 - seq_len - pred_hours
train = df.iloc[:-372, 1:]
test = df.iloc[-372 - seq_len - pred_hours:, 1:]

def profit(pred, pip, total=True):
    ned = pred.cpu().detach().numpy()
    ned = np.where(ned>0.5, 1, -1)
    # ned = np.where(ned> 0.6, 1, ned)
    # ned = np.where(ned< 0.4, -1, ned)
    # ned[(ned != 1)&(ned != -1)] =0
    ap = ned*pip * 10000
    if total:
        return np.sum(ap)
    else:
        return ap

class Forex(Dataset):
    def __init__(self, df, seq_len=100, pred_hours = 1):
        super().__init__()
        self.pred_hours = pred_hours
        self.seq_len = seq_len
        self.df = df.values
        for i in np.argwhere(self.df == 0):
            a, b = i
            self.df[a, b] = self.df[a - 1, b]

    def __len__(self):
        return len(self.df) - self.seq_len - 2 - self.pred_hours

    def __getitem__(self, item):
        std = torch.Tensor([ 1.1960e-01,  1.0929e-01,  1.1639e-01,  1.1580e-01,  2.1910e+03,
                 8.8094e-04,  8.3699e-04,  7.7023e-04,  8.6849e-04,  2.1474e+03,
                 1.0025e-01,  9.1172e-02,  9.3719e-02,  9.9312e-02,  3.0291e+03,
                 9.8328e-04,  9.1960e-04,  9.6045e-04,  9.8505e-04,  2.0675e+03,
                 7.8079e-04,  7.2146e-04,  7.0829e-04,  7.5794e-04,  2.2338e+03,
                 1.3214e-01,  1.2168e-01,  1.3328e-01,  1.3046e-01,  4.9807e+03,
                 1.4860e-01,  1.3303e-01,  1.4739e-01,  1.4396e-01,  2.7807e+03,
                 1.0015e-03,  8.4915e-04,  9.3257e-04,  9.7762e-04,  2.0885e+03,
                 8.0313e-04,  7.6616e-04,  7.1742e-04,  7.7969e-04,  1.5349e+03,
                 1.4785e-03,  1.3635e-03,  1.8143e-03,  1.4648e-03,  2.1705e+03,
                 1.1488e-03,  1.0118e-03,  1.0972e-03,  1.1378e-03,  2.0229e+03,
                 1.1823e-01,  1.0630e-01,  1.1848e-01,  1.1411e-01,  3.2741e+03,
                 1.1122e-03,  1.1408e-03,  1.0701e-03,  1.0888e-03,  3.8207e+03])
        raw = self.df[item:item + self.seq_len]
        a_diff = self.df[item + self.pred_hours:item + self.seq_len + self.pred_hours] - self.df[item:item + self.seq_len]
        self.a_diff = a_diff
        long = np.where(a_diff >= 0, 1, 0)
        # pips = a_diff * self.df[item + self.seq_len - 1]
        norm = raw[1:] - raw[:-1]
        tposed = np.concatenate((np.zeros(norm.shape[1])[None], norm), axis=0) / std
        return tposed, long[-50:,-2], a_diff[-1, -2]


class ForexLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # self.up = nn.Linear(65, 100)
        # self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(65, 65, num_layers=2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(65, 1)
        torch.nn.init.xavier_normal_(self.linear.weight)
        self.sigmoid = torch.nn.Sigmoid()

    def init_hidden(self):
        init = torch.zeros(2, self.batch_size, 65).to(device).double()
        self.hidden = (init, init)

    def forward(self, input, predict=False):
        self.batch_size = input.shape[0]
        self.init_hidden()
        # self.scaled = self.dropout(F.elu(self.up(input)))
        out, self.hidden = self.lstm(input, self.hidden)
        out = out[:, -50:, :]
        return self.sigmoid (self.linear(out).reshape(self.batch_size, 50))


net = ForexLSTM().double().to(device)
if os.path.exists(network_file) & load:
    net.load_state_dict(torch.load('saves/network_lstm.pt'))
    # net.lstm.flatten_parameters()
    print('loaded net')

train_dataset = Forex(train, seq_len=seq_len, pred_hours=pred_hours)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = Forex(test, seq_len=seq_len, pred_hours=pred_hours)
test_loader = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)

criterion = torch.nn.BCELoss()

tloss = []
eloss = []
tacc = []
acc = []

for cycle in range(cycles):
    print(f'{cycle+1}/{cycles}')
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0035, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    start = time.time()
    for epoch in range(epochs):
        peloss = []
        batch_profit = []
        for i, batch in enumerate(train_loader):
            x, y, a = batch
            x, y, a = x.to(device), y.to(device), a.numpy()
            pred = net(x)
            loss = criterion(pred.float(), y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = pred[:, -1]
            with torch.no_grad():
                peloss.append(loss.item())
                batch_profit.append(profit(pred,a))
        with torch.no_grad():
            pred, a = pred[-370:], a[-370:]
            tacc.append(profit(pred, a))
            for set in test_loader:
                x_t, y_t, c = set
                x_t, y_t, c = x_t.to(device), y_t.to(device), c.numpy()
                pred_t = net(x_t)
                loss_t = criterion(pred_t.float(), y_t.float())
                pred_t = pred_t[:, -1]
                tloss.append(loss_t.item())
                acc.append(profit(pred_t, c))
            for param in optimizer.param_groups:
                dict = {'epoch': [f'{epoch+1}/{epochs}'],
                        'Test profit': [f'{round(profit(pred_t, c), 2)} pips'],
                        'Train profit': [f'{round(profit(pred, a), 2)} pips'],
                        'Test loss': [round(loss_t.item(), 4)],
                        'Train loss': [round(np.mean(peloss), 4)],
                        'Learning rate': [param['lr']],
                        'Batch profit': [np.where(np.array(batch_profit) > 0, 1, 0).mean()*100],
                        'ETA': [f'{round(time.time()-start, 2)}/{round((time.time()-start)/(epoch+1)*epochs, 2)}']}
                q = pd.DataFrame.from_dict(dict)
                print('\n', q.to_string(index=False, justify='center'))
        scheduler.step()
        eloss.append(np.mean(peloss))

if os.path.exists(graph_file):
    with open(graph_file, 'rb') as f:
        o_eloss, o_tloss, o_tacc, o_acc = pickle.load(f)
        o_eloss = o_eloss + eloss
        o_tloss = o_tloss + tloss
        o_tacc = o_tacc + tacc
        o_acc = o_acc + acc
else:
    o_eloss, o_tloss, o_tacc, o_acc = eloss, tloss, tacc, acc

plt.show()
plt.plot(o_eloss, color='blue')
plt.plot(o_tloss, color='orange')
plt.title('loss per epoch')
plt.show()
plt.plot(o_acc, color='orange')
plt.plot(o_tacc, color='blue')
plt.title('profit per epoch')
plt.show()
plt.plot(o_acc, color='orange')
plt.show()

ap = profit(pred_t, c, False)
plt.plot([np.sum(ap[:i]) for i in np.arange(len(ap))+1])
plt.title('simulated profit')
plt.show()

# print('pips:', 10000 * np.sum(c.reshape(-1) * np.where(pre > 0.5, 1, 0)))

if save:
    torch.save(net.state_dict(), network_file)
    with open(graph_file, 'wb') as f:
        graph = (o_eloss, o_tloss, o_tacc, o_acc)
        pickle.dump(graph, f)
    print('saved')

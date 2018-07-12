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
network_file = 'saves/network_Qgate.pt'
graph_file = 'saves/graph_Qgate.pkl'
batch_size = 1000
epochs = 20
cycles = 1
seq_len = 200
lr = 0.01
stride = 8

device = torch.device('cuda')
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
df = pd.read_feather('merged.feather')
pred_hours = 4 - 1
# -8302 - seq_len - pred_hours
train = df.iloc[:-372, 1:]
test = df.iloc[-372 - seq_len - pred_hours:, 1:]

convert = torch.DoubleTensor([1, -1]).to(device)
convert.requires_grad = True
def gain(pred, pip):
    # 2*(torch.nn.functional.sigmoid(pip)-0.5)) + 0.1*
    return torch.mean(pred * pip.view(-1))

one = torch.ones(1).to(device).long()
zero = torch.zeros(1).to(device).long()
def profit(pred, pip, total=True, accuracy=False):
    ned = pred.cpu().numpy().reshape(-1, 1)
    pip = pip.cpu().numpy().reshape(-1, 1)
    ap = ned * pip * 10000
    if total:
        return np.sum(ap)
    elif accuracy:
        accuracy = np.mean(np.where(ned > 0,1,-1) == np.where(pip > 0, 1, -1))
        return accuracy
    else:
        return ap

class Forex(Dataset):
    def __init__(self, df, seq_len=100, pred_hours = 1, stride = 1):
        super().__init__()
        self.stride = stride
        self.pred_hours = pred_hours
        self.seq_len = seq_len
        self.df = df.values
        for i in np.argwhere(self.df == 0):
            a, b = i
            self.df[a, b] = self.df[a - 1, b]

    def __len__(self):
        return int((len(self.df) - self.seq_len - 2 - self.pred_hours)/self.stride)

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
        item = self.stride*item
        raw = self.df[item:item + self.seq_len]
        a_diff = self.df[item + self.seq_len + self.pred_hours] - self.df[item + self.seq_len - 1]
        # pips = a_diff * self.df[item + self.seq_len - 1]
        norm = raw[1:] - raw[:-1]
        tposed = np.concatenate((np.zeros(norm.shape[1])[None], norm), axis=0) / std
        return tposed, a_diff[-2]


class ForexLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(65, 100)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(100, 100, num_layers=2, batch_first=True)
        torch.nn.init.xavier_normal_(self.up.weight)
        self.softmax = torch.nn.Softmax(dim=1)
        self.gate = nn.Linear(100, 1)
        torch.nn.init.xavier_normal_(self.gate.weight)
        self.tanh = torch.nn.Tanh()

    def init_hidden(self):
        init = torch.zeros(2, self.batch_size, 100).to(device).double()
        self.hidden = (init, init)

    def forward(self, input):
        self.batch_size = input.shape[0]
        self.init_hidden()
        self.scaled = self.dropout(F.elu(self.up(input)))
        out, self.hidden = self.lstm(self.scaled, self.hidden)
        # out = out+self.scaled
        self.out = out[:, -1, :]
        self.lin = self.gate(self.out).reshape(-1,1)
        self.confidence = self.tanh(self.dropout(self.lin))
        return self.confidence.reshape(-1,1)


net = ForexLSTM().double().to(device)
if os.path.exists(network_file) & load:
    net.load_state_dict(torch.load(network_file))
    # net.lstm.flatten_parameters()
    print('loaded net')

train_dataset = Forex(train, seq_len=seq_len, pred_hours=pred_hours, stride=stride)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
test_dataset = Forex(test, seq_len=seq_len, pred_hours=pred_hours)
test_loader = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False, num_workers=4, pin_memory=False)

criterion = torch.nn.CrossEntropyLoss()


tloss = []
ploss = []
eloss = []
tacc = []
acc = []
for cycle in range(cycles):
    print(f'cycle: {cycle+1}/{cycles}')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    start = time.time()
    std = []
    for epoch in range(epochs):
        batch_profit = []
        peloss = []
        for i, batch in enumerate(train_loader):
            x, a = batch
            x = x.to(device)
            pred = net(x)
            std = a.std()
            d = (a/std).to(device)
            # + 0.04*d)
            loss = gain(pred, d)
            optimizer.zero_grad()
            qloss = -loss
            qloss.backward()
            optimizer.step()
            with torch.no_grad():
                pred, a = pred[-370:], a[-370:]
                ploss.append(loss.item())
                peloss.append(loss.item())
                batch_profit.append(profit(pred, a))
                if loss.item() != loss.item():
                    break
        eloss.append(np.mean(peloss))
        with torch.no_grad():
            tacc.append(profit(pred, a))
            for set in test_loader:
                x_t, c = set
                x_t = x_t.to(device)
                pred_t = net(x_t)
                e = (c/c.std()).to(device)
                loss_t = gain(pred_t, e)
                tloss.append(loss_t.item())
                acc.append(profit(pred_t, c))
            dict = {'epoch': [f'{epoch+1}/{epochs}'],
                    'Test profit': [f'{round(profit(pred_t, c), 2)} pips'],
                    'Train profit': [f'{round(profit(pred, a), 2)} pips'],
                    'Test gain': [round(loss_t.item(), 4)],
                    'Train gain': [round(eloss[-1], 4)],
                    'Test accuracy': [round(profit(pred_t, c, total=False, accuracy=True), 4)],
                    'Train accuracy': [round(profit(pred, a, total=False, accuracy=True), 4)],
                    'Learning rate': [optimizer.param_groups[0]['lr']],
                    'ETA': [f'{round(time.time()-start, 2)}/{round((time.time()-start)/(epoch+1)*epochs, 2)}']}
            q = pd.DataFrame.from_dict(dict)
            print('\n', q.to_string(index=False, justify='center'))
        scheduler.step()

if os.path.exists(graph_file):
    with open(graph_file, 'rb') as f:
        o_ploss, o_eloss, o_tloss, o_tacc, o_acc = pickle.load(f)
        o_ploss = o_ploss + ploss
        o_eloss = o_eloss + eloss
        o_tloss = o_tloss + tloss
        o_tacc = o_tacc + tacc
        o_acc = o_acc + acc
else:
    o_ploss, o_eloss, o_tloss, o_tacc, o_acc = ploss, eloss, tloss, tacc, acc

plt.plot(o_ploss)
plt.title('gain per update')
plt.show()
plt.plot(o_eloss, color='blue')
plt.plot(o_tloss, color='orange')
plt.title('gain per epoch')
plt.show()
plt.plot(o_acc, color='orange')
plt.plot(o_tacc, color='blue')
plt.title('profit per epoch')
plt.show()

ap = profit(pred_t, c, False)
plt.plot([np.sum(ap[:i]) for i in np.arange(len(ap))+1])
plt.title('simulated profit')
plt.show()

# print('pips:', 10000 * np.sum(c.reshape(-1) * np.where(pre > 0.5, 1, 0)))

if save:
    torch.save(net.state_dict(), network_file)
    with open(graph_file, 'wb') as f:
        graph = (o_ploss, o_eloss, o_tloss, o_tacc, o_acc)
        pickle.dump(graph, f)
    print('saved')

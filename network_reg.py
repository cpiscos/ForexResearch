import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import pickle

save = True
network_file = 'saves/network_reg.pt'
graph_file = 'saves/graph_reg.pkl'
batch_size = 900
epochs = 10
seq_len = 400

device = torch.device('cuda')
df = pd.read_feather('merged.feather')
pred_hours = 4 - 1
train = df.iloc[-20302 - seq_len - pred_hours:-302, 1:]
test = df.iloc[-302 - seq_len - pred_hours:, 1:]

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
        a_diff = self.df[item + self.seq_len + self.pred_hours] - self.df[item + self.seq_len - 1]
        actual = a_diff * 10000
        # pips = a_diff * self.df[item + self.seq_len - 1]
        norm = raw[1:] - raw[:-1]
        tposed = np.concatenate((np.zeros(norm.shape[1])[None], norm), axis=0) / std
        return tposed, actual[-2], a_diff[-2]


class ForexLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # self.up = nn.Linear(65, 100)
        # self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(65, 65, num_layers=2, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(65, 1)
        torch.nn.init.xavier_normal_(self.linear.weight)
        # self.sigmoid = torch.nn.Sigmoid()

    def init_hidden(self):
        init = torch.zeros(2, self.batch_size, 65).to(device).double()
        self.hidden = (init, init)

    def forward(self, input):
        self.batch_size = input.shape[0]
        self.init_hidden()
        # self.scaled = self.dropout(F.elu(self.up(input)))
        out, self.hidden = self.lstm(input, self.hidden)
        # out = out+self.scaled
        self.out = out[:, -1, :]
        return self.linear(self.out)


net = ForexLSTM().double().to(device)
if os.path.exists(network_file) & save:
    net.load_state_dict(torch.load(network_file))
    # net.lstm.flatten_parameters()
    print('loaded net')

train_dataset = Forex(train, seq_len=seq_len, pred_hours=pred_hours)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_dataset = Forex(test, seq_len=seq_len, pred_hours=pred_hours)
test_loader = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False, num_workers=4, pin_memory=True)

criterion = torch.nn.MSELoss()

tloss = []
ploss = []
eloss = []
tacc = []
acc = []
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)

def accuracy(pred, y, bar=0, avg=True):
    pred = pred.cpu().detach().numpy().reshape(-1)
    y = y.cpu().detach().numpy().reshape(-1)
    accuracy = np.where(pred > bar, 1, 0) == np.where(y > bar, 1, 0)
    if bar != 0:
        merged = np.concatenate((pred.reshape(1, -1), y.reshape(1, -1)), axis = 0)
        if bar > 0:
            select = np.where(merged > bar, 1, 0)
        else:
            select = np.where(merged < bar, 1, 0)
        select = select[:, select[0] == 1]
        accuracy = select[0] == select[1]
    if avg:
        return np.mean(accuracy)
    else:
        return accuracy

for epoch in tqdm(range(epochs)):
    peloss = []
    for i, batch in enumerate(train_loader):
        scheduler.step()
        x, y, a = batch
        x, y = x.to(device), y.to(device)
        pred = net(x)
        loss = criterion(pred, y.reshape(-1, 1).double())
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
        with torch.no_grad():
            ploss.append(loss.item())
            peloss.append(loss.item())
            if loss.item() != loss.item():
                break
    avg_train_accuracy = accuracy(pred, y, 0)
    tacc.append(avg_train_accuracy)
    with torch.no_grad():
        for set in test_loader:
            x_t, y_t, c = set
            x_t, y_t, c = x_t.to(device), y_t.to(device), c.numpy()
            pred_t = net(x_t)
            loss_t = criterion(pred_t, y_t.reshape(-1, 1).double())
            tloss.append(loss_t.item())
            avg_accuracy = accuracy(pred_t, y_t, 0)
            acc.append(avg_accuracy)
        for param in optimizer.param_groups:
            print('\n', avg_accuracy, avg_train_accuracy, loss_t.item(), loss.item(), param['lr'])
    eloss.append(np.mean(peloss))

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

plt.plot(np.arange(-40, 41), [accuracy(pred_t, y_t, i) for i in np.arange(-40, 41)])
plt.show()
ac = [i for i in np.where(accuracy(pred_t, y_t, 0, avg=False) == True, 1, -1)]
plt.plot([np.sum(ac[:i]) for i in np.arange(len(ac))+1])
plt.show()
plt.plot(o_ploss)
plt.title('loss per update')
plt.show()
plt.plot(o_eloss, color='blue')
plt.plot(o_tloss, color='orange')
plt.title('loss per epoch')
plt.show()
plt.plot(o_acc, color='orange')
plt.plot(o_tacc, color='blue')
plt.title('accuracy')
plt.show()

pre = pred_t.cpu().detach().numpy().reshape(-1)
ap = c.reshape(-1) * np.where(pre > 0, 1, 0)
plt.plot([np.sum(ap[:i]) for i in np.arange(len(ap))+1])
plt.title('profit')
plt.show()

print('pips:', 10000 * np.sum(ap))
print('trading both ways:', 10000 * np.sum(c.reshape(-1) * np.where(pre>0,1,-1)))

if save:
    torch.save(net.state_dict(), network_file)
    with open(graph_file, 'wb') as f:
        graph = (o_ploss, o_eloss, o_tloss, o_tacc, o_acc)
        pickle.dump(graph, f)
    print('saved')

import pandas as pd
import numpy as np
import torch.nn as nn
import torch

df = pd.read_feather('merged.feather').values[:,1:].astype(float)


def update_open(trades, t):
    global df
    for i, v in trades.items():
        trades[i][1] = (df[t,-2] - df[i,-2])*v[0]*10000
    return trades

policy = nn.Sequential(
    nn.Linear(67, 4),
    nn.Softmax(0)).double()

state = []
open_trades = {}
balance = 100
for t in np.arange(len(df))+1:
    state = df[t]-df[t-1]
    open_trades = update_open(open_trades, t)
    equity = np.sum([i[1] for i in open_trades.values()])
    state = np.concatenate((state, np.array([equity]), np.array([balance])))
    action = policy(torch.from_numpy(state))
    action = torch.argmax(action)
    if equity <= -balance or balance <= 0:
        print('\nNo balance\nGame over')
        break
    print('\nOpen Trades:',open_trades)
    info = {'Hour': t,
            'Equity': [equity],
            'Balance': [balance]}
    info = pd.DataFrame.from_dict(info)
    print(info.to_string(index=False))
    if action == 0:
        open_trades[t] = [1, 0]
    elif action == 1:
        open_trades[t] = [-1, 0]
    elif action == 2:
        pass
    else:
        balance += equity
        open_trades = {}
    balance -= 1


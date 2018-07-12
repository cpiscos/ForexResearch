import pandas as pd
import os

data_dir = 'data/'

for i, file in enumerate(sorted(os.listdir(data_dir))):
    raw = pd.read_csv(data_dir + file, parse_dates=[0], index_col=0)
    raw = raw.add_suffix('_' + file[:6])
    if i == 0:
        merged = raw
    else:
        merged = merged.merge(raw, how='outer', left_index=True, right_index=True)
    print(file[:6])
merged.reset_index().to_feather('merged.feather')

import pandas as pd
from  os.path import exists

train_df = pd.read_csv('./data/train.csv')

test_df = pd.read_csv('./data/test.csv')

train_df.to_pickle('./data/train_1.pkl')

test_df.to_pickle('./data/test_1.pkl')

# print(train_df.head())
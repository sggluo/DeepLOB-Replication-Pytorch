import torch
import torch.nn as nn
import polars as pl
from polars import col as c
from torch.utils.data import Dataset
from src.configs.configs import DATA_DIR
import numpy as np

def load_data():
    path1 = DATA_DIR/'train_data.parquet'
    train = pl.scan_parquet(path1)
    train = train.collect().to_numpy()

    path2 = DATA_DIR/'test_data.parquet'
    test = pl.scan_parquet(path2)
    test = test.collect().to_numpy()
    return train, test

def data_processing(train, T=100, D=40, num_classes=3):
    X_train_raw = train[:,:D]
    y_train_raw = train[:,-5:]
    y_train_raw = y_train_raw - 1

    N = X_train_raw.shape[0]

    X_train = np.zeros( (N-T+1,T,D) )
    y_train_h10 = np.zeros( (N-T+1,1) )
    y_train_h20 = np.zeros( (N-T+1,1) )
    y_train_h30 = np.zeros( (N-T+1,1) )
    y_train_h50 = np.zeros( (N-T+1,1) )
    y_train_h100 = np.zeros( (N-T+1,1) )


    # sliding window
    for i in range(N-T+1):
       X_train[i] = X_train_raw[i:i+T,:]
       y_train_h10[i] = y_train_raw[i+T-1,0]
       y_train_h20[i] = y_train_raw[i+T-1,1]
       y_train_h30[i] = y_train_raw[i+T-1,2]
       y_train_h50[i] = y_train_raw[i+T-1,3]
       y_train_h100[i] = y_train_raw[i+T-1,4]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_train = torch.unsqueeze(X_train,1)
    y_train_h10 = torch.tensor( y_train_h10,dtype=torch.long)
    y_train_h20 = torch.tensor( y_train_h20,dtype=torch.long)
    y_train_h30 = torch.tensor( y_train_h30,dtype=torch.long)
    y_train_h50 = torch.tensor( y_train_h50,dtype=torch.long)
    y_train_h100 = torch.tensor( y_train_h100,dtype=torch.long)

    y_train = torch.cat([y_train_h10, y_train_h20, y_train_h30,
                   y_train_h50, y_train_h100], dim=1)


    return X_train, y_train


# Create Pytorch dataset for Deeplob
class DPDataset(Dataset):
    def __init__(self, X, y, h):
        self.len = y.shape[0] # x.shape[0] == y.shape[0]
        self.X = X
        # h =0,1,2,3,4, mapping to labels at horizons of 10, 20, 30, 50, 100 steps
        self.y = y[:,h]

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
      # X: (N,c,sl,n_f)
      return self.X[idx], self.y[idx]
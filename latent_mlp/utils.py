import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):

    def __init__(self, train_x, train_y, prompt):
        self.train_x = train_x
        self.train_y = train_y
        self.prompt = prompt

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return (self.train_x[idx], self.train_y[idx], self.prompt[idx])


def my_collate_fn(batch):
    train_x_path, train_y_path, prompt  = zip(*batch) 
    train_x, train_y= [], []
    for i in range(len(train_x_path)):
        train_x.append(np.load(train_x_path[i]))
        train_y.append(np.load(train_y_path[i]))
    train_x = torch.Tensor(np.array(train_x)).squeeze()
    train_y = torch.Tensor(np.array(train_y)).squeeze()
    return train_x, train_y, prompt
    
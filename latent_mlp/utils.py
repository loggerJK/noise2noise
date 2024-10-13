import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return (self.train_x[idx], self.train_y[idx])


def my_collate_fn(batch):
    train_x_path, train_y_path = zip(*batch) 
    train_x, train_y = [], []
    for i in range(len(train_x_path)):
        train_x.append(torch.load(train_x_path[i]))
        train_y.append(torch.load(train_y_path[i]))
    train_x = torch.concat(train_x)
    train_y = torch.concat(train_y)
    return train_x, train_y
    
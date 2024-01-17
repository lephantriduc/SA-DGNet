import torch
from torch.utils.data import Dataset, DataLoader


class DatasetFromNumpy1(Dataset):
    def __init__(self, x_train, y_train, t_train, e_train, mask1, mask2):
        self.x_train = torch.from_numpy(x_train).float()
        self.t_train = torch.from_numpy(t_train).float()
        self.y_train = torch.from_numpy(y_train).long()
        self.e_train = torch.from_numpy(e_train).long()
        self.mask1 = torch.from_numpy(mask1).float()
        self.mask2 = torch.from_numpy(mask2).float()

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index], self.t_train[index], \
            self.e_train[index], self.mask1[index], self.mask2[index]

    def __len__(self):
        return len(self.y_train)


def make_dataloader1(x, y, t, e, mask1, mask2, batch_size):
    dataset = DatasetFromNumpy1(x, y, t, e, mask1, mask2)
    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=True, num_workers=0)


class DatasetFromNumpy2(Dataset):
    def __init__(self, x_train, t_train, e_train, mask1, mask2):
        self.x_train = torch.from_numpy(x_train).float()
        self.t_train = torch.from_numpy(t_train).float()
        self.e_train = torch.from_numpy(e_train).long()
        self.mask1 = torch.from_numpy(mask1).float()
        self.mask2 = torch.from_numpy(mask2).float()

    def __getitem__(self, index):
        return self.x_train[index], self.t_train[index], \
            self.e_train[index], self.mask1[index], self.mask2[index]

    def __len__(self):
        return len(self.t_train)


def make_dataloader2(x, t, e, mask1, mask2, batch_size):
    dataset = DatasetFromNumpy2(x, t, e, mask1, mask2)
    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=True, num_workers=0)

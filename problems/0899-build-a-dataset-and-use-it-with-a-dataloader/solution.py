import torch
from torch.utils.data import Dataset

class TensorPairDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        # TODO: store X and y on the instance
        self.x=X
        self.y=y

    def __len__(self):
        # TODO: return the number of samples
        return len(self.x)

    def __getitem__(self, idx):
        # TODO: return the (x, y) pair at idx
        return(self.x[idx],self.y[idx])

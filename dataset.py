import torch
from torch.utils import data

class scRNAseqDataset(data.Dataset):
    def __init__(self, data, labels):
        """
        Custom Dataset for single-cell RNA sequencing data.

        Parameters:
        - data (Tensor or ndarray): The input data (e.g., gene expression matrix).
        - labels (Tensor or ndarray): Corresponding labels for classification.
        """
        super().__init__()
        self.data = data
        self.label = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label

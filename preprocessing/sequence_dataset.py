'''
Generates the Dataset object for the DanQ model.
'''
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, dtype={"target": str})

    
    def __len__(self):
        return len(self.data)
    
    def one_hot_encode(self, seq):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        onehot = np.zeros((len(seq), 4), dtype=np.float32)
        for i, base in enumerate(seq):
            if base in mapping:
                onehot[i, mapping[base]] = 1
        return onehot

    def __getitem__(self, idx):
        seq = self.data.iloc[idx]["sequence"]
        target_str = self.data.iloc[idx]["target"]
        
        if not isinstance(target_str, str):
            raise ValueError(f"Expected string for target at index {idx}, got {type(target_str)}: {target_str}")

        onehot = self.one_hot_encode(seq)  # shape (1000, 4)
        target = torch.tensor([int(c) for c in target_str], dtype=torch.float32)

        onehot_tensor = torch.tensor(onehot).permute(1, 0).float()  # Turn into (4, 1000) so 1D conv can be applied
        return onehot_tensor, target

'''
Generates the Dataset object for the DanQ model.
'''
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, csv_file):
      self.data = pd.read_csv(csv_file, dtype=str, keep_default_na=False)

    def __len__(self):
        return len(self.data)
    
    def one_hot_encode(self, seq):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        onehot = np.zeros((len(seq), 4), dtype=np.float32)
        for i, base in enumerate(seq):
            if base in mapping:
                onehot[i, mapping[base]] = 1
        return onehot
    
    def valid(self, seq, target):
        return isinstance(seq, str) and isinstance(target, str) and len(seq) == 1000 and len(target) == 39 and all(c in '01' for c in target)

    def __getitem__(self, idx):
        seq = self.data.iloc[idx]["sequence"]
        target_str = self.data.iloc[idx]["target"]
        
        # Short-term fix for the issue with the target string --> return dummy values
        if not self.valid(seq, target_str):
            print(f"[Warning] Invalid sequence or target at index {idx}. Replacing with a (real) dummy sample")
            seq = 'CCATTTTCCGTGATTTTCAGCTTTCTCGCCATATTCCAGGTCCTACGGTGTGCATTTCTCATTTTTCACGTTTTTCAGTGATTTCAACATTTTTAAAGTCGTCAAGTGGATGTTTCTCATTTTCCATGATTTTCAGTTTTCTTGCCATATTCTATGTCCTACAGTGCACTCTTCTAAATTTTCCACCTTTTTCAGTTTTCCTTGCCATATTTCAAGTGCTAAAGTGTGTATTTCCCATTTTCCGTGATTTTCAGCTTTCTTGCCATATTCCAGGTCCTACAGTGTGCATTTCTCATTTTTCACGTTTTTCAGTGATTTCGTCATTTTTCAAGTCAAGTGGATGTTTCTCATTTTCCATGATTTTCAGTTTTCTTGCCATATTCCATGTCCTACAGTGGACATTTCTAAATTTTCCACCATTTTCAGTTTTCCTCGCCGTATTTCACGTCCTAAAATGTGTATTTCTCTTTTTCCGTGATTTTCAGCTTTCTCGCCATATTCCAGGTCCCACAGTGTGCATTTCTCATTTTTCACGTTTTTCTGTGATTTCGTCATTTTTCAATTCGTCAAGTGGATGTTTCTCATTTTCCATGATTTTCAGTTTTCTTGCCATATTCCATGTCCTAGAGTGGACATTTCTAAATTTTCCACCTTTCTCAGTTTTCCTCGAGATATTTCACGTCCTACAGTGTGTATTTCTCATTTTCCGTCATTTTCTTTTTCTCGCCATATTCAAGGTTCTACAGTGTGCATTTCTCATTCTTCACATTTTTCAGTGATTTCGTCATTTTTCAAGTGGTCAAGTGGATGTTTCTCATTTTCCATGATTTTCAGTTTTCTTACCATATTACATGTCCTACAGTGGACTCTTCTAAATTTTCCACCTTTTTCAGTTTTCCGCGCCATATTTCATGTACTAAATTGTATATTTCTCATTTTCCGTGACTTTCAGTTTTCTCGCCATAATCCAGGTCCTACTGTGTGCATTTCTCATTTTTCA'
            target_str = '000000000000000000000000000000000010000'
        
        onehot = self.one_hot_encode(seq)  # shape (1000, 4)
        target = torch.tensor([int(c) for c in target_str], dtype=torch.float32)

        onehot_tensor = torch.tensor(onehot).permute(1, 0).float()  # Turn into (4, 1000) so 1D conv can be applied
        return onehot_tensor, target

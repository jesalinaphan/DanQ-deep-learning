'''
Defines train and test functions for the DanQ model
'''
import os 
from preprocessing.sequence_dataset import SequenceDataset
from torch.utils.data import random_split, DataLoader

def train():
    pass

def evaluate():
    pass

if __name__ == '__main__':
    data_path = './data/processed.csv'
    assert os.path.exists(data_path), f"{data_path} does not exist. See README on how to genereate {data_path}."
    
    #Loads the dataset
    dataset = SequenceDataset(data_path)

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    #TODO: remove later (for testing only)
    for i, (seq, target) in enumerate(train_loader):
        print(f"Batch {i}:")
        print("Sequence shape:", seq.shape) # (32, 4, 1000) => (batch_size, 4, 1000)
        print("Target shape", target.shape) # (32, 1, 39) => (batch_size, 1, num_peaks)
        if i == 1:
            break

    #TODO: Load the model 

    #TODO: Train 

    #TODO: Evaluate
'''
Defines train and test functions for the DanQ model
'''
import os 
import torch
import numpy as np
from preprocessing.sequence_dataset import SequenceDataset
from torch.utils.data import random_split, DataLoader

def train(model, train, val, criteria, optimizer, epochs, device):
    """Train the DanQ model"""
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for sequences, targets in train:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            #foward and backward pass
            outputs = model(sequences)
            loss = criteria(outputs, targets)
            loss.backward()
            optimizer.step()
            
            #accumilate loss
            train_loss += loss.item() * sequences.size(0)
            
            # calculate accuracy (using 0.5 as threshold)
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == targets).sum().item()
            train_total += targets.numel()
        
        train_loss /= len(train.dataset)
        train_acc = train_correct / train_total
        
        # validation (same)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, targets in val:
                sequences = sequences.to(device)
                targets = targets.to(device)
                outputs = model(sequences)
                loss = criteria(outputs, targets)
                val_loss += loss.item() * sequences.size(0)
                
                # accuracy calculation
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == targets).sum().item()
                val_total += targets.numel()
        
        val_loss /= len(val.dataset)
        val_acc = val_correct / val_total
        
        # save id validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "danq_best_model.pt")
        
        print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    print("training done")

def evaluate(model, test, criteria, device):
    """evaluate the DanQ model"""
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, targets in test:
            sequences = sequences.to(device)
            targets = targets.to(device)
            outputs = model(sequences)
            loss = criteria(outputs, targets)
            test_loss += loss.item() * sequences.size(0)
            
            # calculate accuracy
            predictions = (outputs > 0.5).float()
            test_correct += (predictions == targets).sum().item()
            test_total += targets.numel()
            
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    test_loss /= len(test.dataset)
    test_acc = test_correct / test_total
    
    # combine batch results
    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

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
        print("Target shape", target.shape) # (32, 39) => (batch_size, num_peaks)
        if i == 1:
            break

    #TODO: Load the model 

    #TODO: Train 

    #TODO: Evaluate
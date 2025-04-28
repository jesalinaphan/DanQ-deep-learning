'''
Defines train and test functions for the DanQ model
'''
import os 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from preprocessing.sequence_dataset import SequenceDataset
from torch.utils.data import random_split, DataLoader
from danq import DanQ

def train(model, train_loader, val_loader, device, num_epochs=60, patience=5):
    """
    Train the DanQ model
    """
     
    # loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters())
    
    # variables for early stopping if needed
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass and backward
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # combine loss
            train_loss += loss.item() * sequences.size(0)
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == targets).sum().item()
            train_total += targets.numel()
        
        # Calculate average training loss and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation 
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Print epoch stats
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
              )
        
        # Check for improvement for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            
            # Save the best model
            torch.save(model.state_dict(), 'DanQ_bestmodel.pt')
            print(f'new best model saved')
        else:
            epochs_no_improve += 1
            print(f'No improvement')
            
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered')
                model.load_state_dict(best_model_state)
                break
    
    return model

def evaluate(model, data_loader, criterion=None, device=None):
    """
    Evaluate the DanQ model
    """
    model.eval()
    
    # Initialize criterion and device if not provided
    if criterion is None:
        criterion = nn.BCELoss()
    
    if device is None:
        device = next(model.parameters()).device
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, targets in data_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(sequences)
            
            # Calculate loss
            if criterion is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * sequences.size(0)
            
            predicted = (outputs > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.numel()
    
    # Calculate average loss and accuracy
    if len(data_loader) > 0:
        avg_loss = total_loss / len(data_loader.dataset) if criterion is not None else 0.0
        accuracy = correct / total
    else:
        avg_loss = 0.0
        accuracy = 0.0
    
    return avg_loss, accuracy


if __name__ == '__main__':
    data_path = './data/processed.csv'
    assert os.path.exists(data_path), f"{data_path} does not exist. See README on how to genereate {data_path}."
    
    #Set device, use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Loads the dataset
    dataset = SequenceDataset(data_path)

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=200, num_workers=4)

    #TODO: remove later (for testing only)
    # for i, (seq, target) in enumerate(train_loader):
    #     print(f"Batch {i}:")
    #     print("Sequence shape:", seq.shape) # (32, 4, 1000) => (batch_size, 4, 1000)
    #     print("Target shape", target.shape) # (32, 39) => (batch_size, num_peaks)
    #     if i == 1:
    #         break

    #TODO: Load the model 

    print("loading model")
    model=DanQ().to(device)
    

    #TODO: Train 
    print("Starting training")
    model = train(model, train_loader, val_loader, device, num_epochs=30, patience=3)
    print("Training complete")

    # Evaluate 
    final_loss, final_acc = evaluate(model, val_loader, device=device)
    print(f"Final model - Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), 'DanQ_final_model.pt')
    print("Final model saved")
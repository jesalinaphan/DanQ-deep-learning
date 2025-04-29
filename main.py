import os 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
from preprocessing.sequence_dataset import SequenceDataset
from torch.utils.data import random_split, DataLoader
from danq import DanQ
from tqdm import tqdm
import pandas as pd

def train(model, train_loader, device, criterion, optimizer, num_epochs=60):
    results = {} #dict of results results[epoch] holds 'train_loss', 'labelwise_acc', 'f1_score',

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0

        all_preds = []
        all_targets = []

        for sequences, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            sequences, targets = sequences.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = (outputs > 0.5).float()
            train_correct += (preds == targets).sum().item()
            total_samples += targets.numel()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        labelwise_acc = train_correct / total_samples
        f1 = f1_score(np.vstack(all_targets), np.vstack(all_preds), average='samples')

        results[epoch] = {
            'train_loss': avg_train_loss,
            'labelwise_acc': labelwise_acc,
            'f1_score': f1
        }
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
              f"Label-wise Acc: {labelwise_acc:.2%} | F1-score: {f1:.4f}")

    return model, results

def evaluate(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sequences, targets in tqdm(data_loader, desc=f"Evaluating"):
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            total_correct += (preds == targets).sum().item()
            total_samples += targets.numel()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    labelwise_acc = total_correct / total_samples
    f1 = f1_score(np.vstack(all_targets), np.vstack(all_preds), average='samples')

    print(f"Validation - Loss: {avg_loss:.4f} | Label-wise Acc: {labelwise_acc:.2%} | F1-score: {f1:.4f}")
    return {
        'avg_loss': avg_loss,
        'labelwise_acc': labelwise_acc,
        'f1_score': f1
    }


if __name__ == '__main__':
    data_path = './data/processed.csv'
    assert os.path.exists(data_path), f"{data_path} does not exist. See README on how to generate {data_path}."
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SequenceDataset(data_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=200, num_workers=2)

    model = DanQ().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    checkpoint_path = './checkpoints/danq.pth'
    train_results_path = './results/train_results.csv'
    test_results_path = './results/test_results.csv'


    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Resuming training from saved model")

    print("Starting training...")
    model, results = train(model, train_loader, device, criterion, optimizer, num_epochs=30)
    print("Training complete.")

    #save model checkpoint
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print("Final model saved.")

    #Saving trainig results
    os.makedirs(os.path.dirname(train_results_path), exist_ok=True)
    pd.DataFrame.from_dict(results, orient='index').to_csv(train_results_path)

    print("Evaluating final model...")
    test_results = evaluate(model, val_loader, device, criterion)

    #Saving test results
    os.makedirs(os.path.dirname(test_results_path), exist_ok=True)
    pd.DataFrame.from_dict(test_results, orient='index').to_csv(test_results_path)


    

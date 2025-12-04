import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import os
from src.models.cnn_model import CNN
from src.models.nn_model import MLP
from src.utils.data_loader import get_data_loaders
from src.utils.helpers import set_seed, setup_logger

def train_final_model(hyperparams, model_type='cnn', epochs=10, device=None, save_path='results/final_model.pth'):
    """
    Trains the final model with the best hyperparameters.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    set_seed(42)
    
    # Data
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=int(hyperparams.get('batch_size', 64)),
        pin_memory=(device == 'cuda')
    )
    
    # Model
    if model_type.lower() == 'cnn':
        model = CNN(hyperparams).to(device)
    elif model_type.lower() == 'mlp' or model_type.lower() == 'nn':
        model = MLP(hyperparams).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    lr = float(hyperparams.get('lr', 0.001))
    optimizer_name = hyperparams.get('optimizer', 'adam').lower()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    print(f"Starting final training for {model_type} on {device}...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")

    # Test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    test_acc = correct / total
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to best hyperparams JSON')
    parser.add_argument('--model_type', type=str, default='cnn')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        hyperparams = json.load(f)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_final_model(hyperparams, model_type=args.model_type, epochs=args.epochs, device=device)

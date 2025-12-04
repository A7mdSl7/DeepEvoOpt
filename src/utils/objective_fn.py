import torch
import torch.nn as nn
import torch.optim as optim
import time
from src.models.cnn_model import CNN
from src.models.nn_model import MLP
from src.utils.data_loader import get_data_loaders
from src.utils.helpers import set_seed

def objective_fn(hyperparams, model_type='cnn', epochs=3, device='cpu', data_dir='./data'):
    """
    Unified objective function for hyperparameter optimization.
    
    Args:
        hyperparams (dict): Dictionary of hyperparameters.
        model_type (str): 'cnn' or 'mlp'.
        epochs (int): Number of epochs for short training.
        device (str): 'cpu' or 'cuda'.
        data_dir (str): Path to data directory.
        
    Returns:
        dict: Results containing val_loss, val_acc, time, and hyperparams.
    """
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Prepare data loaders (load once if possible, but for simplicity we load here or rely on caching)
    # Note: In a real scenario, we might want to pass loaders to avoid reloading
    train_loader, val_loader, _ = get_data_loaders(
        batch_size=int(hyperparams.get('batch_size', 64)), 
        data_dir=data_dir, 
        num_workers=0, 
        pin_memory=(device == 'cuda')
    )
    
    # Initialize model
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
        optimizer = optim.Adam(model.parameters(), lr=lr) # Default
        
    # Training Loop
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    training_time = time.time() - start_time
    
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
            
    val_loss /= len(val_loader)
    val_acc = correct / total
    
    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "time": training_time,
        "hyperparams": hyperparams
    }

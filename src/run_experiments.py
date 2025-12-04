import argparse
import json
import os
import pandas as pd
import torch
from src.utils.objective_fn import objective_fn
from src.train import train_final_model

# Import optimizers
from src.optimizers import ga, pso, gwo, aco, firefly, abc, obc_woa, fcr, fcgwo

OPTIMIZERS = {
    'ga': ga,
    'pso': pso,
    'gwo': gwo,
    'aco': aco,
    'firefly': firefly,
    'abc': abc,
    'obc_woa': obc_woa,
    'fcr': fcr,
    'fcgwo': fcgwo
}

DEFAULT_SEARCH_SPACE = {
    'lr': (0.0001, 0.01), # Log scale often better, but linear for simplicity here
    'batch_size': [32, 64, 128],
    'dropout': (0.1, 0.5),
    # CNN specific
    'conv1_out': [16, 32, 64],
    'kernel1': [3, 5],
    'conv2_out': [32, 64, 128],
    'kernel2': [3, 5],
    'fc1_out': [64, 128, 256],
    # MLP specific
    'hidden1': [64, 128, 256, 512],
    'hidden2': [32, 64, 128],
    'optimizer': ['adam', 'sgd']
}

def run_experiment(optimizer_name, model_type, pop_size, max_iter, search_space=None):
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE
        
    print(f"Running {optimizer_name.upper()} on {model_type}...")
    
    optimizer = OPTIMIZERS.get(optimizer_name.lower())
    if not optimizer:
        raise ValueError(f"Optimizer {optimizer_name} not found.")
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    best_solution, history = optimizer.optimize(
        objective_fn,
        pop_size=pop_size,
        max_iter=max_iter,
        search_space=search_space,
        model_type=model_type,
        device=device
    )
    
    # Save results
    os.makedirs('results/logs', exist_ok=True)
    
    # Save best
    with open(f'results/logs/{optimizer_name}_{model_type}_best.json', 'w') as f:
        json.dump(best_solution, f, indent=4)
        
    # Save history
    df = pd.DataFrame(history, columns=['iteration', 'val_loss'])
    df.to_csv(f'results/logs/{optimizer_name}_{model_type}_history.csv', index=False)
    
    print(f"Optimization finished. Best Loss: {history[-1][1]}")
    print(f"Best Hyperparams: {best_solution}")
    
    return best_solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, required=True, help='Optimizer name (ga, pso, etc.) or "all"')
    parser.add_argument('--model_type', type=str, default='cnn', help='cnn or mlp')
    parser.add_argument('--pop_size', type=int, default=10)
    parser.add_argument('--max_iter', type=int, default=5)
    parser.add_argument('--train_final', action='store_true', help='Train final model after optimization')
    
    args = parser.parse_args()
    
    if args.optimizer == 'all':
        optimizers_to_run = list(OPTIMIZERS.keys())
    else:
        optimizers_to_run = [args.optimizer]
        
    for opt in optimizers_to_run:
        best_sol = run_experiment(opt, args.model_type, args.pop_size, args.max_iter)
        
        if args.train_final:
            train_final_model(best_sol, model_type=args.model_type, epochs=10)

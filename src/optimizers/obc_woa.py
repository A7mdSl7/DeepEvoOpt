import random
import numpy as np
import math

def optimize(objective_fn, pop_size=10, max_iter=10, **kwargs):
    """
    Opposition-based Chaotic Whale Optimization Algorithm (OBC-WOA).
    """
    search_space = kwargs.get('search_space', {})
    model_type = kwargs.get('model_type', 'cnn')
    device = kwargs.get('device', 'cpu')
    
    def random_individual():
        individual = {}
        for k, v in search_space.items():
            if isinstance(v, list):
                individual[k] = random.choice(v)
            elif isinstance(v, tuple) and len(v) == 2:
                if isinstance(v[0], int):
                    individual[k] = random.randint(v[0], v[1])
                else:
                    individual[k] = random.uniform(v[0], v[1])
        return individual

    def opposition_based_learning(individual):
        opposite = {}
        for k, v in search_space.items():
            if isinstance(v, tuple):
                opposite[k] = v[0] + v[1] - individual[k]
                if isinstance(v[0], int):
                    opposite[k] = int(opposite[k])
            else:
                opposite[k] = individual[k] # No opposition for categorical
        return opposite

    # Initialize population with OBL
    whales = [random_individual() for _ in range(pop_size)]
    for i in range(pop_size):
        if random.random() < 0.5:
            whales[i] = opposition_based_learning(whales[i])
            
    best_whale = None
    best_score = float('inf')
    
    history = []

    for i in range(max_iter):
        # Evaluate
        for j in range(pop_size):
            res = objective_fn(whales[j], model_type=model_type, device=device)
            loss = res['val_loss']
            
            if loss < best_score:
                best_score = loss
                best_whale = whales[j].copy()
                
        history.append((i, best_score))
        print(f"OBC-WOA Iteration {i+1}/{max_iter}, Best Loss: {best_score:.4f}")
        
        # Chaotic map for parameter p
        # Logistic map: x_{k+1} = 4 * x_k * (1 - x_k)
        # We use it to vary a parameter or just use random
        
        a = 2 - i * (2 / max_iter) # Linearly decreased from 2 to 0
        
        for j in range(pop_size):
            r1 = random.random()
            r2 = random.random()
            
            A = 2 * a * r1 - a
            C = 2 * r2
            
            b = 1
            l = (a - 1) * random.random() + 1
            
            p = random.random()
            
            for k, v in search_space.items():
                if isinstance(v, list):
                    if random.random() < 0.5:
                        whales[j][k] = best_whale[k]
                    else:
                        whales[j][k] = random.choice(v)
                elif isinstance(v, tuple):
                    if p < 0.5:
                        if abs(A) < 1:
                            D = abs(C * best_whale[k] - whales[j][k])
                            new_val = best_whale[k] - A * D
                        else:
                            rand_idx = random.randint(0, pop_size - 1)
                            rand_whale = whales[rand_idx]
                            D = abs(C * rand_whale[k] - whales[j][k])
                            new_val = rand_whale[k] - A * D
                    else:
                        D_prime = abs(best_whale[k] - whales[j][k])
                        new_val = D_prime * math.exp(b * l) * math.cos(2 * math.pi * l) + best_whale[k]
                        
                    # Clip
                    if isinstance(v[0], int):
                        new_val = int(round(new_val))
                        new_val = max(v[0], min(v[1], new_val))
                    else:
                        new_val = max(v[0], min(v[1], new_val))
                    whales[j][k] = new_val

    return best_whale, history

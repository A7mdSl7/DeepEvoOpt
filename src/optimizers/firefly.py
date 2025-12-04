import random
import numpy as np

def optimize(objective_fn, pop_size=10, max_iter=10, **kwargs):
    """
    Firefly Algorithm (FA) for hyperparameter optimization.
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

    fireflies = [random_individual() for _ in range(pop_size)]
    intensities = []
    
    # Evaluate initial
    for f in fireflies:
        res = objective_fn(f, model_type=model_type, device=device)
        intensities.append(res['val_loss']) # Minimize loss -> Light intensity ~ 1/loss
        
    history = []
    
    alpha = 0.5 # Randomness
    beta0 = 1.0 # Attractiveness at r=0
    gamma = 1.0 # Absorption coefficient

    for i in range(max_iter):
        # Sort to find best
        sorted_indices = np.argsort(intensities)
        best_loss = intensities[sorted_indices[0]]
        history.append((i, best_loss))
        print(f"Firefly Iteration {i+1}/{max_iter}, Best Loss: {best_loss:.4f}")
        
        # Move fireflies
        for j in range(pop_size):
            for l in range(pop_size):
                if intensities[l] < intensities[j]: # l is brighter (lower loss) -> j moves to l
                    # Calculate distance
                    dist = 0
                    for k, v in search_space.items():
                        if isinstance(v, tuple): # Numerical distance
                            # Normalize distance
                            range_val = v[1] - v[0]
                            if range_val == 0: range_val = 1
                            d = (fireflies[j][k] - fireflies[l][k]) / range_val
                            dist += d**2
                        else: # Categorical distance (Hamming)
                            if fireflies[j][k] != fireflies[l][k]:
                                dist += 1
                    dist = np.sqrt(dist)
                    
                    beta = beta0 * np.exp(-gamma * dist**2)
                    
                    # Update position
                    for k, v in search_space.items():
                        if isinstance(v, list):
                            if random.random() < beta: # Move towards better categorical
                                fireflies[j][k] = fireflies[l][k]
                            elif random.random() < alpha: # Random move
                                fireflies[j][k] = random.choice(v)
                        elif isinstance(v, tuple):
                            range_val = v[1] - v[0]
                            move = beta * (fireflies[l][k] - fireflies[j][k]) + alpha * (random.random() - 0.5) * range_val
                            new_val = fireflies[j][k] + move
                            
                            # Clip
                            if isinstance(v[0], int):
                                new_val = int(round(new_val))
                                new_val = max(v[0], min(v[1], new_val))
                            else:
                                new_val = max(v[0], min(v[1], new_val))
                            fireflies[j][k] = new_val
                            
        # Evaluate new positions
        for j in range(pop_size):
            res = objective_fn(fireflies[j], model_type=model_type, device=device)
            intensities[j] = res['val_loss']
            
        # Reduce alpha
        alpha *= 0.97

    best_idx = np.argmin(intensities)
    return fireflies[best_idx], history

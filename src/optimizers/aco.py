import random
import numpy as np

def optimize(objective_fn, pop_size=10, max_iter=10, **kwargs):
    """
    Ant Colony Optimization (ACO) for hyperparameter optimization.
    Adapted for continuous/mixed domains (simplified ACOR-like).
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

    # Initialize archive (population)
    archive = [random_individual() for _ in range(pop_size)]
    archive_scores = []
    
    # Evaluate initial population
    for ant in archive:
        res = objective_fn(ant, model_type=model_type, device=device)
        archive_scores.append((res['val_loss'], ant))
        
    # Sort archive
    archive_scores.sort(key=lambda x: x[0])
    
    history = []
    q = 0.5 # Locality of search
    xi = 0.85 # Speed of convergence

    for i in range(max_iter):
        best_loss = archive_scores[0][0]
        history.append((i, best_loss))
        print(f"ACO Iteration {i+1}/{max_iter}, Best Loss: {best_loss:.4f}")
        
        # Generate new solutions
        new_ants = []
        for j in range(pop_size):
            # Select a guide solution (probabilistic selection based on rank)
            # Weights
            weights = [1 / (np.sqrt(2 * np.pi) * q * pop_size) * np.exp(-((k)**2) / (2 * q**2 * pop_size**2)) for k in range(pop_size)]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            
            guide_idx = np.random.choice(range(pop_size), p=probs)
            guide = archive_scores[guide_idx][1]
            
            new_ant = {}
            for k, v in search_space.items():
                if isinstance(v, list):
                    # For categorical, just pick from top k or random mutation
                    if random.random() < 0.8:
                        new_ant[k] = guide[k]
                    else:
                        new_ant[k] = random.choice(v)
                elif isinstance(v, tuple):
                    # Gaussian sampling around guide
                    sigma = xi * abs(archive_scores[-1][1][k] - archive_scores[0][1][k]) / (pop_size) # simplified sigma
                    if sigma == 0: sigma = 0.1 # prevent zero sigma
                    
                    val = random.gauss(guide[k], sigma)
                    
                    # Clip
                    if isinstance(v[0], int):
                        val = int(round(val))
                        val = max(v[0], min(v[1], val))
                    else:
                        val = max(v[0], min(v[1], val))
                    new_ant[k] = val
            new_ants.append(new_ant)
            
        # Evaluate new ants
        new_scores = []
        for ant in new_ants:
            res = objective_fn(ant, model_type=model_type, device=device)
            new_scores.append((res['val_loss'], ant))
            
        # Update archive (keep best pop_size)
        total_scores = archive_scores + new_scores
        total_scores.sort(key=lambda x: x[0])
        archive_scores = total_scores[:pop_size]

    return archive_scores[0][1], history

import random
import numpy as np

def optimize(objective_fn, pop_size=10, max_iter=10, **kwargs):
    """
    Grey Wolf Optimizer (GWO) for hyperparameter optimization.
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

    wolves = [random_individual() for _ in range(pop_size)]
    alpha, beta, delta = None, None, None
    alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')
    
    history = []

    for i in range(max_iter):
        # Evaluate
        for wolf in wolves:
            res = objective_fn(wolf, model_type=model_type, device=device)
            loss = res['val_loss']
            
            if loss < alpha_score:
                alpha_score = loss
                alpha = wolf.copy()
            elif loss < beta_score:
                beta_score = loss
                beta = wolf.copy()
            elif loss < delta_score:
                delta_score = loss
                delta = wolf.copy()
                
        history.append((i, alpha_score))
        print(f"GWO Iteration {i+1}/{max_iter}, Best Loss: {alpha_score:.4f}")
        
        # Update positions
        a = 2 - i * (2 / max_iter) # Linearly decreased from 2 to 0
        
        for j in range(pop_size):
            for k, v in search_space.items():
                if isinstance(v, list): # Categorical - follow alpha mostly
                    if random.random() < 0.5:
                        wolves[j][k] = alpha[k]
                    elif random.random() < 0.5:
                         wolves[j][k] = beta[k]
                    else:
                         wolves[j][k] = delta[k]
                elif isinstance(v, tuple): # Numerical
                    r1, r2 = random.random(), random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[k] - wolves[j][k])
                    X1 = alpha[k] - A1 * D_alpha
                    
                    r1, r2 = random.random(), random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[k] - wolves[j][k])
                    X2 = beta[k] - A2 * D_beta
                    
                    r1, r2 = random.random(), random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[k] - wolves[j][k])
                    X3 = delta[k] - A3 * D_delta
                    
                    new_val = (X1 + X2 + X3) / 3
                    
                    # Clip
                    if isinstance(v[0], int):
                        new_val = int(round(new_val))
                        new_val = max(v[0], min(v[1], new_val))
                    else:
                        new_val = max(v[0], min(v[1], new_val))
                        
                    wolves[j][k] = new_val

    return alpha, history

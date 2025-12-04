import random
import numpy as np

def optimize(objective_fn, pop_size=10, max_iter=10, **kwargs):
    """
    Artificial Bee Colony (ABC) for hyperparameter optimization.
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

    # Initialization
    # Food sources = pop_size / 2
    n_food = pop_size // 2
    food_sources = [random_individual() for _ in range(n_food)]
    fitness = [] # 1 / (1 + loss) if loss > 0 else 1 + abs(loss)
    losses = []
    
    for food in food_sources:
        res = objective_fn(food, model_type=model_type, device=device)
        loss = res['val_loss']
        losses.append(loss)
        fit = 1 / (1 + loss) if loss >= 0 else 1 + abs(loss)
        fitness.append(fit)
        
    limit = 5 # Limit for abandonment
    trials = [0] * n_food
    
    history = []
    best_loss = min(losses)
    best_solution = food_sources[losses.index(best_loss)].copy()

    for i in range(max_iter):
        history.append((i, best_loss))
        print(f"ABC Iteration {i+1}/{max_iter}, Best Loss: {best_loss:.4f}")
        
        # Employed Bees Phase
        for j in range(n_food):
            # Generate new candidate
            k = random.choice(list(search_space.keys()))
            phi = random.uniform(-1, 1)
            
            # Select random partner
            partner_idx = random.choice([idx for idx in range(n_food) if idx != j])
            partner = food_sources[partner_idx]
            
            new_food = food_sources[j].copy()
            v = search_space[k]
            
            if isinstance(v, tuple): # Numerical
                new_val = new_food[k] + phi * (new_food[k] - partner[k])
                if isinstance(v[0], int):
                    new_val = int(round(new_val))
                    new_val = max(v[0], min(v[1], new_val))
                else:
                    new_val = max(v[0], min(v[1], new_val))
                new_food[k] = new_val
            elif isinstance(v, list): # Categorical
                if random.random() < 0.5:
                    new_food[k] = partner[k]
                else:
                    new_food[k] = random.choice(v)
                    
            # Evaluate
            res = objective_fn(new_food, model_type=model_type, device=device)
            new_loss = res['val_loss']
            new_fit = 1 / (1 + new_loss) if new_loss >= 0 else 1 + abs(new_loss)
            
            if new_fit > fitness[j]:
                food_sources[j] = new_food
                fitness[j] = new_fit
                losses[j] = new_loss
                trials[j] = 0
            else:
                trials[j] += 1
                
        # Onlooker Bees Phase
        # Calculate probabilities
        total_fit = sum(fitness)
        probs = [f / total_fit for f in fitness]
        
        for _ in range(n_food):
            # Select food source based on probability
            m = np.random.choice(range(n_food), p=probs)
            
            # Generate new candidate (same logic as employed)
            k = random.choice(list(search_space.keys()))
            phi = random.uniform(-1, 1)
            partner_idx = random.choice([idx for idx in range(n_food) if idx != m])
            partner = food_sources[partner_idx]
            
            new_food = food_sources[m].copy()
            v = search_space[k]
            
            if isinstance(v, tuple):
                new_val = new_food[k] + phi * (new_food[k] - partner[k])
                if isinstance(v[0], int):
                    new_val = int(round(new_val))
                    new_val = max(v[0], min(v[1], new_val))
                else:
                    new_val = max(v[0], min(v[1], new_val))
                new_food[k] = new_val
            elif isinstance(v, list):
                if random.random() < 0.5:
                    new_food[k] = partner[k]
                else:
                    new_food[k] = random.choice(v)
                    
            res = objective_fn(new_food, model_type=model_type, device=device)
            new_loss = res['val_loss']
            new_fit = 1 / (1 + new_loss) if new_loss >= 0 else 1 + abs(new_loss)
            
            if new_fit > fitness[m]:
                food_sources[m] = new_food
                fitness[m] = new_fit
                losses[m] = new_loss
                trials[m] = 0
            else:
                trials[m] += 1
                
        # Scout Bees Phase
        for j in range(n_food):
            if trials[j] > limit:
                food_sources[j] = random_individual()
                res = objective_fn(food_sources[j], model_type=model_type, device=device)
                losses[j] = res['val_loss']
                fitness[j] = 1 / (1 + losses[j]) if losses[j] >= 0 else 1 + abs(losses[j])
                trials[j] = 0
                
        # Update best
        current_best_loss = min(losses)
        if current_best_loss < best_loss:
            best_loss = current_best_loss
            best_solution = food_sources[losses.index(best_loss)].copy()

    return best_solution, history

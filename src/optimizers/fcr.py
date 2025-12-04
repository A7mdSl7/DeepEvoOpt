import random
import numpy as np

def optimize(objective_fn, pop_size=10, max_iter=10, **kwargs):
    """
    Fitness-Centered Recombination (FCR) optimizer.
    A simplified evolutionary strategy focusing on recombination around best solutions.
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

    population = [random_individual() for _ in range(pop_size)]
    history = []
    
    for i in range(max_iter):
        # Evaluate
        scores = []
        for ind in population:
            res = objective_fn(ind, model_type=model_type, device=device)
            scores.append((res['val_loss'], ind))
            
        scores.sort(key=lambda x: x[0])
        best_loss = scores[0][0]
        best_sol = scores[0][1]
        history.append((i, best_loss))
        print(f"FCR Iteration {i+1}/{max_iter}, Best Loss: {best_loss:.4f}")
        
        # Recombination
        # Center around the best solutions
        # Keep top 20%
        top_k = max(1, int(pop_size * 0.2))
        elites = [s[1] for s in scores[:top_k]]
        
        new_pop = elites[:] # Keep elites
        
        while len(new_pop) < pop_size:
            # Pick a random elite
            parent = random.choice(elites)
            child = parent.copy()
            
            # Mutate/Recombine
            for k, v in search_space.items():
                if random.random() < 0.3: # Mutation prob
                    if isinstance(v, list):
                        child[k] = random.choice(v)
                    elif isinstance(v, tuple):
                        # Gaussian mutation around parent
                        sigma = (v[1] - v[0]) * 0.1
                        val = random.gauss(parent[k], sigma)
                        if isinstance(v[0], int):
                            val = int(round(val))
                            val = max(v[0], min(v[1], val))
                        else:
                            val = max(v[0], min(v[1], val))
                        child[k] = val
            new_pop.append(child)
            
        population = new_pop

    return scores[0][1], history

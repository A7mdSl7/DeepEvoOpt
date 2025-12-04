import random
import numpy as np

def optimize(objective_fn, pop_size=10, max_iter=10, **kwargs):
    """
    Genetic Algorithm (GA) for hyperparameter optimization.
    """
    search_space = kwargs.get('search_space', {})
    model_type = kwargs.get('model_type', 'cnn')
    device = kwargs.get('device', 'cpu')
    
    # Helper to generate random individual
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

    # Initialize population
    population = [random_individual() for _ in range(pop_size)]
    history = []
    best_solution = None
    best_fitness = float('inf')

    for i in range(max_iter):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            res = objective_fn(individual, model_type=model_type, device=device)
            loss = res['val_loss']
            fitness_scores.append((loss, individual))
            
            if loss < best_fitness:
                best_fitness = loss
                best_solution = individual.copy()
        
        history.append((i, best_fitness))
        
        # Selection (Tournament)
        selected = []
        for _ in range(pop_size):
            candidates = random.sample(fitness_scores, k=min(3, len(fitness_scores)))
            selected.append(min(candidates, key=lambda x: x[0])[1])
            
        # Crossover
        next_gen = []
        for j in range(0, pop_size, 2):
            p1 = selected[j]
            p2 = selected[(j + 1) % pop_size]
            c1, c2 = p1.copy(), p2.copy()
            
            if random.random() < 0.8: # Crossover prob
                # Uniform crossover
                for k in search_space.keys():
                    if random.random() < 0.5:
                        c1[k], c2[k] = c2[k], c1[k]
            next_gen.extend([c1, c2])
            
        # Mutation
        for j in range(len(next_gen)):
            if random.random() < 0.1: # Mutation prob
                k = random.choice(list(search_space.keys()))
                v = search_space[k]
                if isinstance(v, list):
                    next_gen[j][k] = random.choice(v)
                elif isinstance(v, tuple):
                    if isinstance(v[0], int):
                        next_gen[j][k] = random.randint(v[0], v[1])
                    else:
                        next_gen[j][k] = random.uniform(v[0], v[1])
                        
        population = next_gen[:pop_size]
        print(f"GA Iteration {i+1}/{max_iter}, Best Loss: {best_fitness:.4f}")

    return best_solution, history

import random
import numpy as np

def optimize(objective_fn, pop_size=10, max_iter=10, **kwargs):
    """
    Particle Swarm Optimization (PSO) for hyperparameter optimization.
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

    # Initialize particles
    particles = [random_individual() for _ in range(pop_size)]
    velocities = [{k: 0 for k in search_space} for _ in range(pop_size)]
    pbest = particles[:]
    pbest_scores = [float('inf')] * pop_size
    gbest = None
    gbest_score = float('inf')
    
    history = []
    
    w = 0.5 # Inertia
    c1 = 1.5 # Cognitive
    c2 = 1.5 # Social

    for i in range(max_iter):
        for j in range(pop_size):
            res = objective_fn(particles[j], model_type=model_type, device=device)
            loss = res['val_loss']
            
            if loss < pbest_scores[j]:
                pbest_scores[j] = loss
                pbest[j] = particles[j].copy()
                
            if loss < gbest_score:
                gbest_score = loss
                gbest = particles[j].copy()
        
        history.append((i, gbest_score))
        print(f"PSO Iteration {i+1}/{max_iter}, Best Loss: {gbest_score:.4f}")
        
        # Update velocities and positions
        for j in range(pop_size):
            for k, v in search_space.items():
                # Handle categorical/discrete variables by simple random choice if velocity is high?
                # Or just standard PSO for continuous/integer
                
                if isinstance(v, list): # Categorical - Random reset if "velocity" suggests change
                     if random.random() < 0.1: # Mutation-like
                        particles[j][k] = random.choice(v)
                elif isinstance(v, tuple): # Numerical
                    r1, r2 = random.random(), random.random()
                    
                    current_val = particles[j][k]
                    pbest_val = pbest[j][k]
                    gbest_val = gbest[k]
                    
                    new_vel = w * velocities[j][k] + c1 * r1 * (pbest_val - current_val) + c2 * r2 * (gbest_val - current_val)
                    velocities[j][k] = new_vel
                    
                    new_val = current_val + new_vel
                    
                    # Clip
                    if isinstance(v[0], int):
                        new_val = int(round(new_val))
                        new_val = max(v[0], min(v[1], new_val))
                    else:
                        new_val = max(v[0], min(v[1], new_val))
                        
                    particles[j][k] = new_val

    return gbest, history

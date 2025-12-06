import random
import numpy as np
from math import inf
import copy

def optimize(objective_fn, pop_size=10, max_iter=10, **kwargs):
    """
    Fuzzy-Controlled GWO (FCGWO) - corrected version.
    Uses simple fuzzy-like rules to adapt the parameter 'a'.
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
            else:
                # fallback: try uniform between 0 and 1
                individual[k] = random.random()
        return individual

    # initialize population
    wolves = [random_individual() for _ in range(pop_size)]
    alpha, beta, delta = None, None, None
    alpha_score, beta_score, delta_score = inf, inf, inf
    
    history = []

    for i in range(max_iter):
        evals = []  # collect (fitness, individual, res)
        for wolf in wolves:
            res = objective_fn(wolf, model_type=model_type, device=device)
            loss = res.get('val_loss', None)
            if loss is None:
                # if objective didn't return val_loss, skip or treat as high cost
                loss = float('inf')
            evals.append((loss, wolf, res))

        # sort by fitness ascending (lower is better)
        evals.sort(key=lambda x: x[0])

        # assign alpha, beta, delta safely from sorted list (if available)
        if len(evals) >= 1:
            alpha_score, alpha, alpha_res = evals[0][0], copy.deepcopy(evals[0][1]), evals[0][2]
        if len(evals) >= 2:
            beta_score, beta, beta_res = evals[1][0], copy.deepcopy(evals[1][1]), evals[1][2]
        else:
            beta_score, beta = inf, copy.deepcopy(alpha) if alpha is not None else random_individual()
        if len(evals) >= 3:
            delta_score, delta, delta_res = evals[2][0], copy.deepcopy(evals[2][1]), evals[2][2]
        else:
            delta_score, delta = inf, copy.deepcopy(beta) if beta is not None else (copy.deepcopy(alpha) if alpha is not None else random_individual())

        history.append((i, alpha_score))
        print(f"FCGWO Iteration {i+1}/{max_iter}, Best Loss: {alpha_score:.4f}")

        # diversity metric
        fitness_values = [e[0] for e in evals]
        diversity = np.std(fitness_values) if len(fitness_values) > 0 else 0.0

        # Base 'a' decreases linearly
        base_a = 2 - i * (2 / max_iter)

        # Fuzzy-like adaptation
        if diversity < 0.01:
            a = min(2.0, base_a * 1.5)
        elif diversity > 0.5:
            a = base_a * 0.5
        else:
            a = base_a

        # Update wolves
        for j in range(pop_size):
            for k, v in search_space.items():
                # ensure we have numeric placeholders for alpha/beta/delta at key k
                ak = alpha.get(k) if alpha is not None and k in alpha else None
                bk = beta.get(k) if beta is not None and k in beta else None
                dk = delta.get(k) if delta is not None and k in delta else None

                # discrete choices
                if isinstance(v, list):
                    # if any leader missing, fallback to random choice
                    if ak is None or bk is None or dk is None:
                        wolves[j][k] = random.choice(v)
                    else:
                        r = random.random()
                        if r < 0.33:
                            wolves[j][k] = ak
                        elif r < 0.66:
                            wolves[j][k] = bk
                        else:
                            wolves[j][k] = dk

                elif isinstance(v, tuple) and len(v) == 2:
                    # numeric range update using GWO formulas
                    # ensure leaders values are numeric
                    # if any leader missing, use current value as fallback
                    cur_val = wolves[j].get(k, random.uniform(v[0], v[1]) if not isinstance(v[0], int) else random.randint(v[0], v[1]))
                    a1_r1, a1_r2 = random.random(), random.random()
                    A1 = 2 * a * a1_r1 - a
                    C1 = 2 * a1_r2
                    alpha_k = ak if ak is not None else cur_val
                    D_alpha = abs(C1 * alpha_k - cur_val)
                    X1 = alpha_k - A1 * D_alpha

                    a2_r1, a2_r2 = random.random(), random.random()
                    A2 = 2 * a * a2_r1 - a
                    C2 = 2 * a2_r2
                    beta_k = bk if bk is not None else cur_val
                    D_beta = abs(C2 * beta_k - cur_val)
                    X2 = beta_k - A2 * D_beta

                    a3_r1, a3_r2 = random.random(), random.random()
                    A3 = 2 * a * a3_r1 - a
                    C3 = 2 * a3_r2
                    delta_k = dk if dk is not None else cur_val
                    D_delta = abs(C3 * delta_k - cur_val)
                    X3 = delta_k - A3 * D_delta

                    new_val = (X1 + X2 + X3) / 3.0

                    # clamp and round if integer
                    if isinstance(v[0], int):
                        new_val = int(round(new_val))
                        new_val = max(v[0], min(v[1], new_val))
                    else:
                        new_val = max(v[0], min(v[1], new_val))

                    wolves[j][k] = new_val

                else:
                    # fallback: if search_space spec unknown, randomize a bit
                    wolves[j][k] = wolves[j].get(k, random.random())

    return alpha, history

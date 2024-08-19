import numpy as np
import pandas as pd  # Adicionando a importação do pandas
from funcao_custo import funcao_custo

# Função de mutação personalizada (exemplo: DE/rand/1)
def mutacao_rand_1(population, F, rng):
    r1, r2, r3 = rng.choice(population.shape[0], 3, replace=False)
    return population[r1] + F * (population[r2] - population[r3])

# Estratégia DE personalizada
def custom_differential_evolution(strategy, bounds, track, popsize=20, maxiter=100, tol=1e-9, F=0.8, seed=None):
    rng = np.random.default_rng(seed)
    dimensions = len(bounds)
    population = np.zeros((popsize, dimensions))

    for i in range(dimensions):
        population[:, i] = rng.uniform(bounds[i][0], bounds[i][1], size=popsize)

    costs = np.array([funcao_custo(ind, track=track) for ind in population])
    best_idx = np.argmin(costs) 
    best = population[best_idx]

    log = []

    for iteration in range(maxiter):
        print(f"Iteração {iteration} / {maxiter}")

        next_population = np.zeros_like(population)
        for i in range(popsize):
            vi = mutacao_rand_1(population, F, rng)

            cross_points = rng.uniform(size=dimensions) < 0.9
            if not np.any(cross_points):
                cross_points[rng.integers(0, dimensions)] = True
            next_population[i] = np.where(cross_points, vi, population[i])

            next_population[i] = np.clip(next_population[i], [b[0] for b in bounds], [b[1] for b in bounds])

        next_costs = np.array([funcao_custo(ind, track=track) for ind in next_population])
        for i in range(popsize):
            if next_costs[i] < costs[i]:  # Minimizar a função de custo
                population[i] = next_population[i]
                costs[i] = next_costs[i]
                if costs[i] < costs[best_idx]:
                    best_idx = i
                    best = population[i]

        if iteration % 5 == 0 or iteration == maxiter - 1:
            mean_cost = np.mean(costs)
            log.append({
                'Iteration': iteration,
                'Best k': best[0],
                'Best c': best[1],
                'Best Cost': costs[best_idx],
                'Mean Cost': mean_cost
            })
            print(
                f"Iteração {iteration}: Melhor k={best[0]}, Melhor c={best[1]}, Melhor Custo={costs[best_idx]}, Custo Médio={mean_cost}")

        if np.std(costs) < tol:
            print(f"Convergência alcançada na iteração {iteration}.")
            break

    df = pd.DataFrame(log)
    df.to_excel('optimization_log.xlsx', index=False)

    return best, costs[best_idx]

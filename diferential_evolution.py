import numpy as np
from typing import Callable
from numpy.typing import NDArray

def evolucao_diferencial_adaptativa(
    func: Callable[[list], float],
    bounds: NDArray[np.int_],
    pop_size=110,
    F=0.8,
    CR=0.9,
    max_generations=1000,
):
    population = [
        np.array([np.random.randint(bound[0], bound[1]) for bound in bounds])
        for _ in range(pop_size)
    ]

    results = np.array([func(solution) for solution in population])
    num_params = len(bounds)

    for gen in range(max_generations):
        print(f"Geração {gen}", end="\r")
        for i in range(pop_size):
            indices = list(range(pop_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)

            diff = population[b] - population[c]
            trial = population[i] + F * diff
            trial = np.clip(trial, bounds[:, 0], bounds[:, 1])  # enforce bounds
            trial = np.round(trial).astype(int)  # round to integers

            j_rand = np.random.randint(num_params)
            trial_vec = np.array(
                [
                    (
                        trial[j]
                        if np.random.rand() < CR or j == j_rand
                        else population[i][j]
                    )
                    for j in range(num_params)
                ]
            )

            trial_fitness = func(trial_vec)
            if np.all(trial_fitness < results[i]):
                population[i] = trial_vec
                results[i] = trial_fitness

        F = np.clip(F * np.random.rand(), 0.4, 1.2)
        CR = np.clip(CR * np.random.rand(), 0.1, 1.0)

        if np.std(results) < 1e-10:
            print(f"Convergência alcançada na geração {gen}.")
            break

        gen += 1
        results = np.where(np.isfinite(results), results, np.inf)
        best_idx = np.argmin(results)
        best_params = population[best_idx]
        best_fitness = results[best_idx]
        best_params = np.clip(best_params, bounds[:, 0], bounds[:, 1])
        standard_deviation = np.std(results)

    return best_params, best_fitness, standard_deviation

import numpy as np

# Evolução Diferencial Auto-Adaptativa (EDA)
def evolucao_diferencial_adaptativa(func, bounds, pista, pop_size=130, F=0.8, CR=0.9, max_generations=100000000000):
    dim = len(bounds)
    pop = np.random.rand(pop_size, dim)
    pop_denorm = bounds[:, 0] + pop * (bounds[:, 1] - bounds[:, 0])
    fitness = np.array([func(ind, pista) for ind in pop_denorm])

    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    log_tempo_voltas = []  # Armazenar o melhor tempo de cada geração
    log_tempo_voltas.append(fitness[best_idx])

    gen = 0
    while gen < max_generations:
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            r1, r2, r3 = pop[np.random.choice(idxs, 3, replace=False)]
            mutante = r1 + F * (r2 - r3)
            mutante = np.clip(mutante, 0, 1)
            cruzado = np.where(np.random.rand(dim) < CR, mutante, pop[i])
            cruzado_denorm = bounds[:, 0] + cruzado * (bounds[:, 1] - bounds[:, 0])
            f = func(cruzado_denorm, pista)
            if f < fitness[i]:
                fitness[i] = f
                pop[i] = cruzado
                if f < fitness[best_idx]:
                    best_idx = i
                    best = cruzado_denorm

        # Auto-adaptação de F e CR
        F = np.clip(F * np.random.rand(), 0.4, 1.2)
        CR = np.clip(CR * np.random.rand(), 0.1, 1.0)

        if np.std(fitness) < 1e-10:
            print(f"Convergência alcançada na geração {gen}.")
            break
        gen += 1

    return best, fitness[best_idx]

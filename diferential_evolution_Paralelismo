import numpy as np
from typing import Callable
from numpy.typing import NDArray
from Pista import TRACK
from joblib import Parallel, delayed


def customDifferentialEvolution(
    func: Callable[[list, int, int, float], float],
    bounds: NDArray[np.int_],
    pop_size=400,
    F=0.8,
    CR=0.9,
    max_generations=100,
    track_start=0,
    track_end=200,
    gensToPlot=[100, 200, 300],
    folder="charts",
    use_parallel=True,
    n_jobs=5,   # número de núcleos em paralelo
    strategy="rand/1/bin",  # <<< NOVO parâmetro
    checkpoint_step=1
):
    checkpoints = pick_checkpoints(track_start, track_end, checkpoint_step)
    waypoints_population = np.random.randint(bounds[0][0], bounds[0][1], (pop_size, track_end - track_start))
    max_speed = bounds[1][1]

    # Avaliação inicial
    if use_parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(func)(waypoints_population[i], checkpoints, max_speed)
            for i in range(pop_size)
        )
    else:
        results = [func(waypoints_population[i], checkpoints, max_speed)
                   for i in range(pop_size)]
    lap_times = [result[0].astype(float) for result in results]
    
    best_results = []
    std_deviations = []

    for gen in range(max_generations):
        if gen in gensToPlot:
            best_idx = np.argmin(lap_times)
            path = results[best_idx][1]
            speed_profile = results[best_idx][2]
            TRACK.plotar_tracado_na_pista_com_velocidade(
                f"{folder}/tracado geração {gen} - teste continuo.png",
                path, speed_profile, track_start, track_end
            )

        print(f"Geração {gen + 1}", end="\r")

        best_lap_index = np.argmin(lap_times)
        best_vector = waypoints_population[best_lap_index]

        std_dev = np.std(lap_times)
        best_results.append(lap_times[best_lap_index])
        std_deviations.append(std_dev)

        if std_dev < 0.1:
            print(f"Convergência alcançada na geração {gen}.")
            break

        offspring_population = []
        for i in range(pop_size):
            parent = waypoints_population[i]
            trial_vector = [
                np.round(val).astype(int)
                for val in generateTrialVector(
                    waypoints_population, i, F, bounds[0], best_vector, strategy
                )
            ]
            offspring = generateOffspring(trial_vector, parent, CR)
            offspring_population.append((offspring, i))

        # Avaliar descendentes
        if use_parallel:
            offspring_results = Parallel(n_jobs=n_jobs)(
                delayed(func)(offspring, checkpoints, max_speed)
                for offspring, _ in offspring_population
            )
        else:
            offspring_results = [func(offspring, checkpoints, max_speed)
                                 for offspring, _ in offspring_population]

        # Atualizar população
        for (offspring, i), (offspring_result, offspring_path, offspring_speeds) in zip(
            offspring_population, offspring_results
        ):
            if offspring_result < lap_times[i]:
                waypoints_population[i] = offspring
                lap_times[i] = offspring_result
                results[i] = [offspring_result, offspring_path, offspring_speeds]

    best_idx = np.argmin(lap_times)
    best_waypoints = waypoints_population[best_idx] # <<< AINDA É UM ARRAY DE FLOATS
    best_fitness = lap_times[best_idx]
    path = results[best_idx][1]
    speed_profile = results[best_idx][2]
    standard_deviation = np.std(lap_times)
    best_waypoints = np.array(best_waypoints, dtype=int)
    num_pontos_por_checkpoint = len(TRACK.CHECKPOINTS[0])
    best_waypoints = np.clip(best_waypoints, 0, num_pontos_por_checkpoint - 1)
    
    TRACK.plotar_tracado_na_pista_com_velocidade(
        f"{folder}/tracado geração {gen} - teste continuo.png",
        path, speed_profile, track_start, track_end, checkpoints=True
    )

    return (best_waypoints, speed_profile), best_fitness, standard_deviation, (best_results, std_deviations)


def generateTrialVector(population, parent_index, F, bounds, best_vector, strategy):
    pop_size = len(population)

    if strategy == "rand/1/bin":
        # target = random, mutação com diferença de 2 aleatórios
        indices = list(range(pop_size))
        indices.remove(parent_index)
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        target_vector = population[r1]
        trial_vector = target_vector + F * (population[r2] - population[r3])

    elif strategy == "best/1/bin":
        # target = melhor, mutação com diferença de 2 aleatórios
        indices = list(range(pop_size))
        indices.remove(parent_index)
        r1, r2 = np.random.choice(indices, 2, replace=False)
        trial_vector = best_vector + F * (population[r1] - population[r2])

    elif strategy == "current-to-best/1/bin":
        # target = pai atual, direcionado ao melhor
        indices = list(range(pop_size))
        indices.remove(parent_index)
        r1, r2 = np.random.choice(indices, 2, replace=False)
        parent = population[parent_index]
        trial_vector = parent + F * (best_vector - parent) + F * (population[r1] - population[r2])

    else:
        raise ValueError(f"Estratégia {strategy} não reconhecida.")

    trial_vector = np.clip(trial_vector, bounds[0], bounds[1])
    return trial_vector


def generateOffspring(trial_vector, parent, CR):
    return np.array(
        [
            trial_vector[j] if np.random.rand() < CR else parent[j]
            for j in range(len(parent))
        ]
    )


def pick_checkpoints(track_start: int, track_end: int, step: int):
    checkpoints = []
    for i in range(track_start, track_end, step):
        if i + step > track_end:
            i = track_end
        checkpoints.append(i)

    return checkpoints

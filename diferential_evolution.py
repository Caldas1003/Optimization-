import numpy as np
from typing import Callable
from numpy.typing import NDArray
from PistaComAtrito import TRACK
from Chassis_metamodel import FullModel
from path import PATH

def customDifferentialEvolution(
    func_objective: Callable[[float, float, float, float], tuple[float, float]],
    bounds: NDArray[NDArray[float]],
    pop_size=400,
    F=0.8,
    CR=0.9,
    max_generations=100,
    track_start=0,
    track_end=100,
    gensToPlot=[100, 200, 300],
    folder="charts"
):
    population = np.array([
        [
            np.random.uniform(bounds[0][0], bounds[0][1]), # distribution
            np.random.uniform(bounds[1][0], bounds[1][1]), # Kds
            np.random.uniform(bounds[2][0], bounds[2][1]), # Kts
            np.random.uniform(bounds[3][0], bounds[3][1]), # Kpds
            np.random.uniform(bounds[4][0], bounds[4][1]), # Kpts
            np.random.uniform(bounds[5][0], bounds[5][1]), # Kf
            np.random.uniform(bounds[6][0], bounds[6][1]), # Kt
            np.random.uniform(bounds[7][0], bounds[7][1]), # Cds
            np.random.uniform(bounds[8][0], bounds[8][1]), # Cts
            np.random.uniform(bounds[9][0], bounds[9][1]), # W
            np.random.uniform(bounds[10][0], bounds[10][1]), # Lt
            np.random.uniform(bounds[11][0], bounds[11][1]), # Ld
        ]
        for i in range(pop_size)
    ])

    max_speed = bounds[12][1]
    results = np.array([
        func_objective(
            max_speed, 
            *FullModel(*individual).get_acceleration_limits()
        ) for individual in population
    ], dtype=object)
    lap_times = [result[0].astype(float) for result in results]
    
    best_results = []
    std_deviations = []

    for gen in range(max_generations):
        if gen in gensToPlot:
            best_idx = np.argmin(lap_times)
            speed_profile = results[best_idx][1]
            TRACK.plotar_tracado_na_pista_com_velocidade(f"{folder}/tracado geração {gen} - teste continuo.png", PATH, speed_profile, track_start, track_end)

        print(f"Geração {gen + 1}", end="\r")

        best_lap_index = np.argmin(lap_times)

        std_dev = np.std(lap_times)

        best_results.append(lap_times[best_lap_index])
        std_deviations.append(std_dev)

        if std_dev < 0.1:
            print(f"Convergência alcançada na geração {gen}.")
            break

        for i in range(pop_size):
            parent = population[i]

            trial_vector = generateTrialVector(population, i, F, bounds)

            offspring = generateOffspring(trial_vector, parent, CR)

            offspring_result, offspring_speeds = func_objective(
                max_speed,
                *FullModel(*offspring).get_acceleration_limits(),
            )
            offspring_is_fitter = np.all(offspring_result < lap_times[i])
            if offspring_is_fitter:
                population[i] = offspring
                lap_times[i] = offspring_result
                results[i] = [offspring_result, offspring_speeds]

    best_idx = np.argmin(lap_times)
    best_parameters = population[best_idx]
    best_fitness = lap_times[best_idx]
    speed_profile = results[best_idx][1]
    standard_deviation = np.std(lap_times)
    TRACK.plotar_tracado_na_pista_com_velocidade(f"{folder}/tracado geração {max_generations} - teste continuo.png", PATH, speed_profile, track_start, track_end)

    return (np.array(best_parameters), np.array(speed_profile)), best_fitness, standard_deviation, (best_results, std_deviations)


def generateTrialVector(population, parent_index, F, bounds):
    target_vector, random_individual_1, random_individual_2 = get_random_vectors(population, parent_index, len(population))

    trial_vector = target_vector + (
        F * (random_individual_1 - random_individual_2)
    )

    trial_vector = np.clip(
        trial_vector, bounds[:12, 0], bounds[:12, 1]
    )  # enforce bounds

    return trial_vector


def get_random_vectors(population, parent_index, population_size):
    indices = list(range(population_size))
    indices.remove(parent_index)
    random_index_1, random_index_2, random_index_3 = np.random.choice(
        indices, 3, replace=False
    )
    target_vector = population[random_index_1]
    random_individual_1 = population[random_index_2]
    random_individual_2 = population[random_index_3]

    return target_vector, random_individual_1, random_individual_2


def generateOffspring(trial_vector, parent, CR):
    return np.array(
        [
            (
                trial_vector[j]
                if np.random.rand() < CR
                else parent[j]
            )
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
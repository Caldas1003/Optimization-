import numpy as np
from typing import Callable
from numpy.typing import NDArray
from PistaComAtrito import TRACK


def customDifferentialEvolution(
    func: Callable[[list], float],
    bounds: NDArray[np.int_],
    pop_size=400,
    F=0.8,
    CR=0.9,
    max_generations=100,
    track_size=200,
    gensToPlot=[100, 200, 300],
    folder="charts"
):
    # middle_of_track = np.ones(200, dtype=int) * 46
    waypoints_population = np.random.randint(bounds[0][0], bounds[0][1], (pop_size, track_size))
    # waypoints_population[pop_size // 2] = middle_of_track  

    max_speed = bounds[1][1]
    results = np.array([func(waypoints_population[i], max_speed) for i in range(pop_size)], dtype=object)
    lap_times = [result[0].astype(float) for result in results]
    
    best_results = []
    std_deviations = []

    for gen in range(max_generations):
        if gen in gensToPlot:
            best_idx = np.argmin(lap_times)
            best_waypoints = waypoints_population[best_idx]
            speed_profile = results[best_idx][1]
            TRACK.plotar_tracado_na_pista_com_velocidade(f"{folder}/tracado geração {gen} - teste continuo.png", best_waypoints, speed_profile, track_size)

        print(f"Geração {gen + 1}", end="\r")

        best_lap_index = np.argmin(lap_times)

        std_dev = np.std(lap_times)

        best_results.append(lap_times[best_lap_index])
        std_deviations.append(std_dev)

        if std_dev < 0.1:
            print(f"Convergência alcançada na geração {gen}.")
            break

        for i in range(pop_size):
            waypoints_parent = waypoints_population[i]

            waypoints_trial_vector = [np.round(waypoint_index).astype(int) for waypoint_index in generateTrialVector(waypoints_population, i, F, bounds[0])]

            waypoints_offspring = generateOffspring(waypoints_trial_vector, waypoints_parent, CR)

            offspring_result, offspring_profile = func(waypoints_offspring, max_speed)
            offspring_is_fitter = np.all(offspring_result < lap_times[i])
            if offspring_is_fitter:
                waypoints_population[i] = waypoints_offspring
                lap_times[i] = offspring_result
                results[i][1] = offspring_profile

    best_idx = np.argmin(lap_times)
    best_waypoints = waypoints_population[best_idx]
    best_fitness = lap_times[best_idx]
    speed_profile = results[best_idx][1]
    standard_deviation = np.std(lap_times)
    TRACK.plotar_tracado_na_pista_com_velocidade(f"{folder}/tracado geração {max_generations} - teste continuo.png", best_waypoints, speed_profile, track_size)

    return (best_waypoints, speed_profile), best_fitness, standard_deviation, (best_results, std_deviations)


def generateTrialVector(population, parent_index, F, bounds):
    target_vector, random_individual_1, random_individual_2 = get_random_vectors(population, parent_index, len(population))

    trial_vector = target_vector + (
        F * (random_individual_1 - random_individual_2)
    )

    trial_vector = np.clip(
        trial_vector, bounds[0], bounds[1]
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
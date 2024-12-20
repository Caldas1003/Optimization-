import numpy as np
from typing import Callable
from numpy.typing import NDArray


def customDifferentialEvolution(
    func: Callable[[list], float],
    bounds: NDArray[np.int_],
    pop_size=90,
    F=0.8,
    CR=0.9,
    max_generations=1000,
):
    

    middle_of_track = np.ones(200, dtype=int) * 46
    waypoints_population = np.random.randint(bounds[0][0], bounds[0][1], (pop_size, 200))
    waypoints_population[pop_size // 2] = middle_of_track  
    speeds_population = np.random.uniform(bounds[1][0], 3, (pop_size, 200))

    results = np.array([func(waypoints_population[i], speeds_population[i]) for i in range(pop_size)])
    lap_times = [result[0].astype(float) for result in results]
    furthest_points_achieved = [result[1].astype(int) for result in results]
    
    # reasons = [result[2] for result in results]
    
    best_results = []
    std_deviations = []
    furthest_points_achieved_per_gen = []
    best_result_is_the_furthest = []

    for gen in range(max_generations):
        print(f"Geração {gen + 1}", end="\r")

        best_lap_index = np.argmin(lap_times)
        furthest_point_index = np.argmax(furthest_points_achieved)

        std_dev = np.std(lap_times)

        best_result_is_the_furthest.append(best_lap_index == furthest_point_index)
        best_results.append(lap_times[best_lap_index])
        furthest_points_achieved_per_gen.append(furthest_points_achieved[furthest_point_index])
        std_deviations.append(std_dev)

        if std_dev < 0.1:
            print(f"Convergência alcançada na geração {gen}.")
            break

        for i in range(pop_size):
            waypoints_parent = waypoints_population[i]
            speeds_parent = speeds_population[i]

            waypoints_trial_vector = [np.round(waypoint_index).astype(int) for waypoint_index in generateTrialVector(waypoints_population, i, F, bounds[0])]
            speeds_trial_vector = generateTrialVector(speeds_population, i, F, bounds[1])

            waypoints_offspring = generateOffspring(waypoints_trial_vector, waypoints_parent, CR)
            speeds_offspring = generateOffspring(speeds_trial_vector, speeds_parent, CR)

            offspring_result, fpa_offspring, reason_for_failure = func(waypoints_offspring, speeds_offspring)
            offspring_is_fitter = np.all(offspring_result < lap_times[i])
            if offspring_is_fitter:
                waypoints_population[i] = waypoints_offspring
                speeds_population[i] = speeds_offspring
                lap_times[i] = offspring_result
                furthest_points_achieved[i] = fpa_offspring
                # reasons.append(reason_for_failure)

        gen += 1

    best_idx = np.argmin(lap_times)
    best_waypoints = waypoints_population[best_idx]
    best_speeds = speeds_population[best_idx]
    best_fitness = lap_times[best_idx]
    standard_deviation = np.std(lap_times)

    return (best_waypoints, best_speeds), best_fitness, standard_deviation, (best_results, std_deviations, furthest_points_achieved_per_gen, best_result_is_the_furthest)


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
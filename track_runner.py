import numpy as np
import time
from PistaComAtrito import TRACK
from matplotlib import pyplot as plt
from typing import Callable
from numpy.typing import NDArray


def calculate_angle(P1, P2, P3):
    vector_A = np.array(P2) - np.array(P1)
    vector_B = np.array(P3) - np.array(P2)
    
    scalar_product = np.dot(vector_A, vector_B)
    magnitude_A = np.linalg.norm(vector_A)
    magnitude_B = np.linalg.norm(vector_B)
    
    angle_in_rad = np.arccos(scalar_product / (magnitude_A * magnitude_B))
    
    return np.degrees(angle_in_rad)


def build_gates() -> list[list]:
    X_malha, Y_malha, Z_malha = TRACK.gerar_malha()
    X_left = X_malha[0][0]
    X_right = X_malha[0][1]
    Y_left = Y_malha[0][0]
    Y_right = Y_malha[0][1]
    Z_left = Z_malha[0][0]
    Z_right = Z_malha[0][1]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    gates = list()
    for i, value in enumerate(X_left):
        left_point = (X_left[i], Y_left[i], Z_left[i])
        right_point = (X_right[i], Y_right[i], Z_right[i])

        num_points = 100
        t = np.linspace(0, 1, num_points)
        waypoints = np.outer(1 - t, left_point) + np.outer(
            t, right_point
        )  # 100 points between the left and right points

        gates.append(waypoints)
        # ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2])

    # plt.savefig("gates.png")
    return gates


GATES = build_gates()


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
            trial = np.clip(trial, bounds[:, 0], bounds[:, 1]) # enforce bounds
            trial = np.round(trial).astype(int) # round to integers

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


def total_distance(waypoints: list) -> float:
    total_distance = 0

    for i, value in enumerate(waypoints):
        last_waypoint = waypoints[i - 1]
        current_waypoint = waypoints[i]
        distance_between_waypoints = np.linalg.norm(
            GATES[i][current_waypoint] - GATES[i - 1][last_waypoint]
        )
        # distance_between_waypoints = np.sqrt((current_waypoint[0] - last_waypoint[0])**2 + (current_waypoint[1] - last_waypoint[1])**2 + (current_waypoint[2] - last_waypoint[2])**2)
        total_distance += distance_between_waypoints

    return total_distance


def run():
    start_time = time.time()

    best_params, best_fitness, standard_deviation = evolucao_diferencial_adaptativa(total_distance, np.array([(0, 99) for _ in range(100)]), max_generations=1000)

    time_elapsed = time.time() - start_time
    hours, seconds_remain = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(seconds_remain, 60)

    print(f"Melhores parâmetros encontrados: {best_params}")
    print(f"Menor distância total encontrada: {best_fitness}")
    print(f"Desvio padrão: {standard_deviation}")
    print(f"Total time elapsed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")


run()

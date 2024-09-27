import numpy as np
import time
from checkpoint_builder import CHECKPOINTS
from diferential_evolution import evolucao_diferencial_adaptativa


def calculate_angle(P1, P2, P3):
    vector_A = np.array(P2) - np.array(P1)
    vector_B = np.array(P3) - np.array(P2)
    
    scalar_product = np.dot(vector_A, vector_B)
    magnitude_A = np.linalg.norm(vector_A)
    magnitude_B = np.linalg.norm(vector_B)
    
    angle_in_rad = np.arccos(scalar_product / (magnitude_A * magnitude_B))
    
    return np.degrees(angle_in_rad)



def total_distance(waypoints: list) -> float:
    total_distance = 0

    for i, value in enumerate(waypoints):
        last_waypoint = waypoints[i - 1]
        current_waypoint = waypoints[i]
        distance_between_waypoints = np.linalg.norm(
            CHECKPOINTS[i][current_waypoint] - CHECKPOINTS[i - 1][last_waypoint]
        )
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

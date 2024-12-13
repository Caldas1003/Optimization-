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


def get_turn_radius(P1, P2, P3):
    p2_p3_len = np.sqrt((P3 - P2)**2)
    p1_p3_len = np.sqrt((P3 - P1)**2)
    p1_p2_len = np.sqrt((P2 - P1)**2)

    stretch_1 = P2 - P1
    stretch_2 = P3 - P2
    vector_product = (
        stretch_1[1] * stretch_2[2] - stretch_1[2] * stretch_2[1],
        stretch_1[2] * stretch_2[0] - stretch_1[0] * stretch_2[2],
        stretch_1[0] * stretch_2[1] - stretch_1[1] * stretch_2[0]
    )
    area = np.sqrt(vector_product[0]**2 + vector_product[1]**2 + vector_product[2]**2) / 2

    if area == 0:
        return 0
     
    return (p2_p3_len * p1_p3_len * p1_p2_len) / (4 * area)


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

def lap_time(lap: list) -> tuple:
    g = 9.8 # gravity (9.8 m/s²)
    allowed_centrifugal_force = 1.5 * g
    furthest_point_achieved = 0
    total_time = 0

    number_of_points = len(lap)
    
    for i in range(number_of_points):
        available_percentage = 1

        previous_point, previous_speed = lap[i - 1]
        current_point, current_speed = lap[i]
        following, following_speed = lap[i + 1]

        turn_radius = get_turn_radius(previous_point, current_point, following)

        if turn_radius > 0:
            centrifugal_force = (current_speed**2) / turn_radius
            if centrifugal_force > allowed_centrifugal_force:
                furthest_point_achieved = i
                break

            available_percentage = 1 - (centrifugal_force / allowed_centrifugal_force)

        gained_speed = current_speed > following_speed
        maximum_allowed_acceleration = 0
        if gained_speed:
            maximum_engine_acceleration = 0.9*g if current_speed < 15.28 else (272195/66227)*current_speed - (12250/66277)*(current_speed**2) # 15.28 m/s = 55 kph
            maximum_allowed_acceleration = maximum_engine_acceleration * available_percentage
        else:
            maximum_braking_power = - 1.5 * g
            maximum_allowed_acceleration = maximum_braking_power * available_percentage

        necessary_acceleration = (following_speed**2 - current_speed**2) / (2*np.linalg.norm(previous_point - current_point))
        if necessary_acceleration > maximum_allowed_acceleration:
            furthest_point_achieved = i
            break

        total_time += (following_speed - current_speed) / necessary_acceleration

    return total_time + ((number_of_points + 1)/(furthest_point_achieved + 1))*100000, furthest_point_achieved


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

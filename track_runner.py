import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from checkpoint_builder import CHECKPOINTS
from diferential_evolution import customDifferentialEvolution


def calculate_angle(P1, P2, P3):
    vector_A = np.array(P2) - np.array(P1)
    vector_B = np.array(P3) - np.array(P2)
    
    scalar_product = np.dot(vector_A, vector_B)
    magnitude_A = np.linalg.norm(vector_A)
    magnitude_B = np.linalg.norm(vector_B)
    
    angle_in_rad = np.arccos(scalar_product / (magnitude_A * magnitude_B))
    
    return np.degrees(angle_in_rad)


def get_turn_radius(P1, P2, P3):
    p2_p3_len = np.linalg.norm(P3 - P2)
    p1_p3_len = np.linalg.norm(P3 - P1)
    p1_p2_len = np.linalg.norm(P2 - P1)

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

def lapTime(waypoints: list, speeds: list) -> tuple:
    number_of_points = len(waypoints)
    
    if number_of_points != len(speeds):
        raise ValueError("Number of speeds must be equal to number of waypoints")
    
    g = 9.8 # gravity (9.8 m/s²)
    allowed_centrifugal_force = 1.5 * g
    furthest_checkpoint_achieved = number_of_points - 1
    total_time = 0
    reason_for_failure = "none"
    
    for i in range(number_of_points):
        if i == number_of_points - 1:
            break # the last and first point are the same, so if we reached the checkpoint 199, we are done

        available_percentage = 1

        previous = i - 1
        current = i
        following = i + 1

        previous_point = CHECKPOINTS[previous][waypoints[previous]]
        current_point = CHECKPOINTS[current][waypoints[current]]
        following_point = CHECKPOINTS[following][waypoints[following]]

        current_speed = speeds[current]
        following_speed = speeds[following]

        turn_radius = get_turn_radius(previous_point, current_point, following_point)

        if turn_radius > 0:
            centrifugal_force = (current_speed**2) / turn_radius
            if centrifugal_force > allowed_centrifugal_force:
                furthest_checkpoint_achieved = current
                reason_for_failure = "turn_speed"
                break

            available_percentage = 1 - (centrifugal_force / allowed_centrifugal_force)

        gained_speed = following_speed > current_speed
        maximum_allowed_acceleration = 0
        if gained_speed:
            maximum_engine_acceleration = 0.9*g if current_speed < 15.28 else (272195/66227)*current_speed - (12250/66277)*(current_speed**2) # 15.28 m/s = 55 kph
            maximum_allowed_acceleration = maximum_engine_acceleration * available_percentage
        else:
            maximum_braking_power = 1.5 * g
            maximum_allowed_acceleration = maximum_braking_power * available_percentage

        stretch_distance = np.linalg.norm(following_point - current_point)
        
        necessary_acceleration = np.abs((following_speed**2 - current_speed**2) / (2*stretch_distance))
    
        if necessary_acceleration > maximum_allowed_acceleration:
            furthest_checkpoint_achieved = current
            reason_for_failure = "acceleration"
            break

        try:
            total_time += (following_speed - current_speed) / necessary_acceleration if necessary_acceleration != 0 else stretch_distance / current_speed
        except:
            print(f"following_speed: {following_speed}")
            print(f"current_speed: {current_speed}")
            print(f"necessary_acceleration: {necessary_acceleration}")

    return total_time + (((number_of_points + 1)/(furthest_checkpoint_achieved + 1)) - 1)*500000, furthest_checkpoint_achieved, reason_for_failure


def run():
    start_time = time.time()
    F = 0.5
    CR = 0.85
    max_gen = 100
    pop_size = 4000

    best_params, best_fitness, standard_deviation, data = customDifferentialEvolution(lapTime, [[0, 99], [0.278, 22.22]], max_generations=max_gen, F=F, CR=CR, pop_size=pop_size)

    time_elapsed = time.time() - start_time
    hours, seconds_remain = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(seconds_remain, 60)

    print(f"Melhores parâmetros encontrados: {best_params}")
    print(f"Menor tempo total encontrada: {best_fitness}")
    print(f"Desvio padrão: {standard_deviation}")
    print(f"Ponto mais longe: {data[2][len(data[2]) - 1]}")
    print(f"Total time elapsed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")


    generations = [i + 1 for i in range(len(data[0]))]
    plt.figure(figsize=(18, 10))
    plt.suptitle(f"Tempo gasto: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\nDesvio padrão: {standard_deviation:.2f}\nMenor tempo: {best_fitness:.2f}\nPonto mais longe alcançado: {data[2][len(data[2]) - 1]}\n", fontsize=16)
    plt.subplots_adjust(wspace=0.8, hspace=1.2)

    plt.subplot(4, 1, 1)
    plt.plot(generations, data[0])
    plt.xlabel("Geração")
    plt.ylabel("Melhor tempo de volta")
    plt.ylim(0 , 2500)
    plt.xlim(0, max_gen)

    plt.subplot(4, 1, 2)
    plt.plot(generations, data[1])
    plt.xlabel("Geração")
    plt.ylabel("Desvio Padrão")
    plt.ylim(0, 10)
    plt.xlim(0, max_gen)

    plt.subplot(4, 1, 3)
    plt.plot(generations, data[2])
    plt.xlabel("Geração")
    plt.ylabel("Ponto mais longe alcançado")
    plt.xlim(0, max_gen)

    # categories, count = np.unique(data[4], return_counts=True)
    # print(categories)
    # print(count)
    # plt.subplot(4, 1, 4)
    # plt.bar(categories, count)
    # plt.xlabel("Motivo da falha")
    # plt.ylabel("Quantidade")
    # plt.xticks(np.arange(len(categories)), categories)
    # plt.subplot(2, 1, 4).plot(generations, data[3], ylabel="Desvio Padrão", xlabel="Geração")

    timestamp = datetime.now().strftime("%H%M %d-%m-%Y")
    plt.savefig(f"charts/{timestamp} pop_size={pop_size} max_gen={max_gen} F={F} CR={CR}.png")

run()

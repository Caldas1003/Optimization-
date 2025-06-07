import numpy as np
import matplotlib.pyplot as plt
import time
import os
import warnings

from datetime import datetime

from PistaComAtrito import TRACK
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
        stretch_1[0] * stretch_2[1] - stretch_1[1] * stretch_2[0],
    )
    area = (
        np.sqrt(
            vector_product[0] ** 2 + vector_product[1] ** 2 + vector_product[2] ** 2
        )
        / 2
    )

    if area == 0:
        return 0

    return (p2_p3_len * p1_p3_len * p1_p2_len) / (4 * area)

def total_distance(waypoints: list) -> float:
    total_distance = 0

    for i, value in enumerate(waypoints):
        last_waypoint = waypoints[i - 1]
        current_waypoint = waypoints[i]
        distance_between_waypoints = np.linalg.norm(
            TRACK.CHECKPOINTS[i][current_waypoint]
            - TRACK.CHECKPOINTS[i - 1][last_waypoint]
        )
        total_distance += distance_between_waypoints

    return total_distance

def get_stretch_distance(P1, P2):
    '''
    Calculate the distance between two points

    P2 is the point that the car is going to,
    P1 is the point that the car is coming from
    '''
    stretch_distance = np.linalg.norm(P2 - P1)
    
    return np.abs(stretch_distance)

def acceleration_available(speed: float) -> float:
    g = 9.8 # gravity
    return 1.5*g if speed < 15.28 else ((-1.5*g*(speed - 15.28)/6.94) + 1.5*g) # 15.28 m/s = 55 kph

def generate_speed_profile(waypoints:list, max_speed: float) -> list:
    speed_profile = []
    number_of_points = len(waypoints)
    max_speed = np.float64(max_speed)

    g = 9.8  # gravity (9.8 m/s²)
    max_centrifugal_force = 1.5 * g
    max_braking = - 1.5 * g

    # first run
    for i in range(number_of_points):
        if i == 0 or i == number_of_points - 1:
            speed_profile.append(max_speed)
            continue

        previous = i - 1
        current = i
        following = i + 1

        previous_point = TRACK.CHECKPOINTS[previous][waypoints[previous]]
        current_point = TRACK.CHECKPOINTS[current][waypoints[current]]
        following_point = TRACK.CHECKPOINTS[following][waypoints[following]]
        
        turn_radius = get_turn_radius(previous_point, current_point, following_point)
        # print(f"prev: {waypoints[previous]}")
        # print(f"curr: {waypoints[current]}")
        # print(f"follw: {waypoints[following]}")
        # print(f"radius: {turn_radius}")

        if turn_radius > 0:
            max_turn_speed = np.sqrt(max_centrifugal_force * turn_radius)
            choosen_speed = max_turn_speed if max_turn_speed < max_speed else max_speed
            speed_profile.append(choosen_speed)
        else:
            speed_profile.append(max_speed)
            
    
    # second run
    for i in range(number_of_points - 1, -1, -1):
        if i == number_of_points - 1:
            continue

        current = i
        previous = i + 1

        current_point = TRACK.CHECKPOINTS[current][waypoints[current]]
        previous_point = TRACK.CHECKPOINTS[previous][waypoints[previous]]

        if speed_profile[current] > speed_profile[previous]:
            stretch_distance = get_stretch_distance(previous_point, current_point)
            # print(f"stretch_distance: {stretch_distance}")
            # print(f"max_braking: {max_braking}")
            # print(f"speed_profile[previous]: {speed_profile[previous]}")
            # print(f"speed_profile[current]: {speed_profile[current]}")
            speed = np.sqrt(speed_profile[previous]**2 - 2*stretch_distance*max_braking)
            if speed < 0:
                print("NEGATIVE SPEED")
            speed_profile[current] = speed if speed < max_speed else max_speed

    # third run
    for i in range(number_of_points):
        if i == 0:
            continue

        current = i
        previous = i - 1

        current_point = TRACK.CHECKPOINTS[current][waypoints[current]]
        previous_point = TRACK.CHECKPOINTS[previous][waypoints[previous]]

        if speed_profile[current] > speed_profile[previous]:
            stretch_distance = get_stretch_distance(previous_point, current_point)
            # print(f"stretch_distance: {stretch_distance}")
            # print(f"max_braking: {max_braking}")
            # print(f"speed_profile[previous]: {speed_profile[previous]}")
            # print(f"speed_profile[current]: {speed_profile[current]}")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                acc = acceleration_available(speed_profile[previous])

                if acc > 1.5*9.8:
                    print(f"acc: {acc}")
                    print(f"speed prev: {speed_profile[previous]}")
                    print(f"speed curr: {speed_profile[current]}")

                speed = np.sqrt(speed_profile[previous]**2 + 2*stretch_distance*acc)
                # print(f"acc: {acc}")
                # print(f"dist: {stretch_distance}")
                # print(f"speed_prev: {speed_profile[previous]}")
                # print(f"speed: {speed}")

                if w:
                    print(f"Warning: {w[0].message}")
                    print(f"stretch_distance: {stretch_distance}")
                    print(f"acc: {acc}")
                    print(f"speed_profile[previous]: {speed_profile[previous]}")
                    print(f"speed_profile[current]: {speed_profile[current]}")
                
                speed_profile[current] = speed if speed < max_speed else max_speed

    return speed_profile

def calculate_time(waypoints: list, speed_profile: list) -> float:
    if len(waypoints) != len(speed_profile):
        raise ValueError("Number of waypoints and speeds must be equal")
    
    number_of_points = len(waypoints)
    total_time = 0

    for i in range(number_of_points):
        if i == 0 or i == number_of_points - 1:
            continue

        current = i
        following = i + 1

        current_point_index = waypoints[current]
        following_point_index = waypoints[following]
        current_point = TRACK.CHECKPOINTS[current][current_point_index]
        following_point = TRACK.CHECKPOINTS[following][following_point_index]
        current_speed = speed_profile[current]
        following_speed = speed_profile[following]

        stretch_distance = get_stretch_distance(current_point, following_point)
        acceleration = (following_speed**2 - current_speed**2) / (2*stretch_distance)

        stretch_time = (following_speed - current_speed) / acceleration if acceleration != 0 else stretch_distance / current_speed
        
        # print(f"acc: {acceleration}")
        # print(f"dist: {stretch_distance}")
        # print(f"curr speed: {current_speed}")
        # print(f"folw speed: {following_speed}")
        # print(f"stretch_time: {stretch_time}")

        total_time += stretch_time
        # print(f"total: {total_time}")

        if stretch_time < 0:
            print(following_speed)
    
    return total_time

def lapTime(waypoints: list, max_speed: float) -> list:
    speed_profile = generate_speed_profile(waypoints, max_speed)

    return [calculate_time(waypoints, speed_profile), speed_profile]

def generations_to_plot(amount: int, step: int) -> list:
    return np.arange(1, amount + 1) * step

def run():
    start_time = time.time()
    F = 0.5
    CR = 0.85
    max_gen = 50000
    pop_size = 1400
    track_size = 70
    gens_to_plot = generations_to_plot(100, 500)

    timestamp = datetime.now().strftime("%Y-%m-%d %H%M")
    folder = f"{timestamp} - speed profile gen"
    os.makedirs(folder, exist_ok=True)

    best_params, best_fitness, standard_deviation, data = customDifferentialEvolution(
        lapTime,
        [[0, 99], [0.278, 22.22]],
        max_generations=max_gen,
        F=F,
        CR=CR,
        pop_size=pop_size,
        track_size=track_size,
        gensToPlot=gens_to_plot,
        folder=folder,
    )

    time_elapsed = time.time() - start_time
    hours, seconds_remain = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(seconds_remain, 60)

    print(f"Melhores parâmetros encontrados: {best_params}")
    print(f"Menor tempo total encontrada: {best_fitness}")
    print(f"Desvio padrão: {standard_deviation}")
    print(f"Total time elapsed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    generations = [i + 1 for i in range(len(data[0]))]
    plt.figure(figsize=(18, 10))
    plt.suptitle(
        f"Tempo gasto: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\nDesvio padrão: {standard_deviation:.2f}\nMenor tempo: {best_fitness:.2f}\n",
        fontsize=16,
    )
    plt.subplots_adjust(wspace=0.8, hspace=1.2)

    plt.subplot(4, 1, 1)
    plt.plot(generations, data[0])
    plt.xlabel("Geração")
    plt.ylabel("Melhor tempo de volta")
    # plt.ylim(0 , 2500)
    # plt.xlim(0, max_gen)

    plt.subplot(4, 1, 2)
    plt.plot(generations, data[1])
    plt.xlabel("Geração")
    plt.ylabel("Desvio Padrão")
    plt.ylim(0, 10)
    # plt.xlim(0, max_gen)

    fileName = f"{timestamp} pop_size={pop_size} max_gen={max_gen} F={F} CR={CR}"

    plt.savefig(f"{folder}/charts.png")
    # TRACK.plotar_tracado_na_pista(f"somulação {timstamp}/{fileName} tracado.png", best_params[0], track_size)


if __name__ ==   '__main__':
    run()

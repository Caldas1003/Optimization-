from datetime import datetime
import matplotlib.pyplot as plt
import time
import numpy as np
import warnings
import os

from PistaComAtrito import TRACK
from diferential_evolution import customDifferentialEvolution


def spline_equation(t: float, a: float, b: float, c: float, d: float) -> float:
    return a*(t**3) + b*(t**2) + c*t + d

def first_derivative(t: float, a: float, b: float, c: float) -> float:
    return 3*a*(t**2) + 2*b*t + c

def second_derivative(t: float, a: float, b: float) -> float:
    return 6*a*t + 2*b

def calculate_coeficients(
    spline_at_t0: float,
    spline_at_t1: float,
    spline_first_derivative_at_t0: float,
    spline_first_derivative_at_t1: float,
) -> dict:
    d = spline_at_t0
    c = spline_first_derivative_at_t0
    b = 3*(spline_at_t1 - spline_at_t0) - 2*spline_first_derivative_at_t0 - spline_first_derivative_at_t1
    a = -2*(spline_at_t1 - spline_at_t0) + spline_first_derivative_at_t0 + spline_first_derivative_at_t1

    return {"a": a, "b": b, "c": c, "d": d}

def catmull_rom_tangent(P_k_plus_1: float, P_k_minus_1: float, u_k_plus_1: float, u_k_minus_1: float, tau = 0.25):
    """k is relative to the Knot (point) you want the parameter u for"""
    return (1 - tau)*(P_k_plus_1 - P_k_minus_1)/(u_k_plus_1 - u_k_minus_1)

def parameter_u(P_k_minus_1: float, P_k: float, u_k_minus_1: float, alpha = 0.5):
    """k is relative to the Knot (point) you want the parameter u for"""
    return u_k_minus_1 + np.abs(P_k - P_k_minus_1)**alpha

def calculate_parameters_u(waypoints: list, track_start: int, track_end: int) -> list:
    u = list()

    for i in range(track_start, track_end):
        u_x_k = 0
        u_y_k = 0

        if i > track_start:
            X_k_minus_1, Y_k_minus_1, Z_k_minus_1 = TRACK.CHECKPOINTS[i - 1][waypoints[i - track_start - 1]]
            X_k, Y_k, Z_k = TRACK.CHECKPOINTS[i][waypoints[i - track_start]]
            u_x_k = parameter_u(X_k_minus_1, X_k, u[i - track_start - 1][0])
            u_y_k = parameter_u(Y_k_minus_1, Y_k, u[i - track_start - 1][1])

        u.append([u_x_k, u_y_k])

    return u

def build_splines(waypoints: list, track_start: int, track_end: int) -> list:
    splines = list()
    u = calculate_parameters_u(waypoints, track_start, track_end)
    for i in range(track_start, track_end - 1):							
        first_derivative_at_t0 = [0, 0]
        first_derivative_at_t1 = [0, 0]

        X_k, Y_k, Z_k = TRACK.CHECKPOINTS[i][waypoints[i - track_start]]
        X_k_plus_1, Y_k_plus_1, Z_k_plus_1 = TRACK.CHECKPOINTS[i + 1][waypoints[i + 1 - track_start]]
        if i < track_end - 2:
            X_k_plus_2, Y_k_plus_2, Z_k_plus_2 = TRACK.CHECKPOINTS[i + 2][waypoints[i + 2 - track_start]]

            u_x_k = u[i - track_start][0]
            u_y_k = u[i - track_start][1]
            u_x_k_plus_2 = u[i - track_start + 2][0]
            u_y_k_plus_2 = u[i - track_start + 2][1]

            first_derivative_at_t1 = [
                catmull_rom_tangent(X_k_plus_2, X_k, u_x_k_plus_2, u_x_k),
                catmull_rom_tangent(Y_k_plus_2, Y_k, u_y_k_plus_2, u_y_k)
            ]

        if i > track_start:
            previous_spline = splines[i - 1 - track_start]
            X_spline, Y_spline = previous_spline
            # t = 1 because the first derivative of the current segment at t=0 should be equal to the first derivative of the previous segment at t=1, same goes for seconds
            first_derivative_at_t0 = [
                    first_derivative(1, X_spline["a"], X_spline["b"], X_spline["c"]),
                    first_derivative(1, Y_spline["a"], Y_spline["b"], Y_spline["c"]),
            ]

        # print(f"i: {i}")
        # print(f"X_k = {X_k}, Y_k = {Y_k}")
        # print(f"X_k_plus_1 = {X_k_plus_1}, Y_k = {Y_k_plus_1}")
        # print(f"first derivatives (t=0) = {first_derivative_at_t0}")
        # print(f"first derivatives (t=1) = {first_derivative_at_t1}")
        # print("---------------------------------------------------------------")

        X_coefs = calculate_coeficients(X_k, X_k_plus_1, first_derivative_at_t0[0], first_derivative_at_t1[0])
        Y_coefs = calculate_coeficients(Y_k, Y_k_plus_1, first_derivative_at_t0[1], first_derivative_at_t1[1])
            
        splines.append([X_coefs, Y_coefs])
    
    return splines

def generate_path(waypoints: list, track_start: int, track_end: int) -> list:
    splines = build_splines(waypoints, track_start, track_end)
    # for spline in splines:
    #      print(spline)
    path = []

    for i in range(track_start, track_end - 1):
        X_spline, Y_spline = splines[i - track_start]
        if i == track_start:
            x = spline_equation(0, X_spline["a"], X_spline["b"], X_spline["c"], X_spline["d"])
            y = spline_equation(0, Y_spline["a"], Y_spline["b"], Y_spline["c"], Y_spline["d"])
            path.append([x, y])
            
                
        for t in np.arange(0.1, 1.1, 0.1):
            x = spline_equation(t, X_spline["a"], X_spline["b"], X_spline["c"], X_spline["d"])
            y = spline_equation(t, Y_spline["a"], Y_spline["b"], Y_spline["c"], Y_spline["d"])
            path.append([x, y])
    
    return np.array(path)

def get_turn_radius(P1, P2, P3, two_dim = True):
	if two_dim:
		P1 = np.append(P1, 0)
		P2 = np.append(P2, 0)
		P3 = np.append(P3, 0)

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

def generate_speed_profile(points:list, max_speed: float) -> list:
    speed_profile = []
    number_of_points = len(points)
		
    max_speed = np.float64(max_speed)

    g = 9.8  # gravity (9.8 m/s²)
    max_centrifugal_force = 1.5 * g
    max_braking = - 1.5 * g

    # first run
    for i in range(number_of_points):
        if i == 0 or i == len(points) - 1:
            speed_profile.append(max_speed)
            continue

        previous = i - 1
        current = i
        following = i + 1

        previous_point = points[previous]
        current_point = points[current]
        following_point = points[following]
        
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

        current_point = points[current]
        previous_point = points[previous]

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

        current_point = points[current]
        previous_point = points[previous]

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

def calculate_angle(P1, P2, P3):
    """
    Calculate the angle between three points P1, P2, and P3.
    The angle calculated is between the extended line of P1P2 with P2P3,
    that is why there is a 180 degree subtraction, the angle we want to use is
    the inside angle between the two vectors P1P2 and P2P3.
    """
    vector_A = np.array(P2) - np.array(P1)
    vector_B = np.array(P3) - np.array(P2)

    scalar_product = np.dot(vector_A, vector_B)
    magnitude_A = np.linalg.norm(vector_A)
    magnitude_B = np.linalg.norm(vector_B)

    angle_in_rad = np.arccos(scalar_product / (magnitude_A * magnitude_B))
    if np.degrees(angle_in_rad) > 180:
        print("Ângulo maior que 180 graus, o que não é esperado.")

    return 180 - np.degrees(angle_in_rad)

def calculate_angle_penalty(P1, P2, P3):
    """
    Penaliza curvas com ângulos pequenos.
    Quanto menor o ângulo, maior a penalização.
    """
    angle = calculate_angle(P1, P2, P3)
    penalty = (5/(90**2))*((180 - angle)**2)

    if penalty < 0:
        print("Penalidade menor que zero, o que não é esperado.")
        print(f"Ângulo: {angle}")

    return penalty

def calculate_time(points: list, speed_profile: list) -> float:
    if len(points) != len(speed_profile):
        raise ValueError("Number of points and speeds must be equal")
    
    number_of_points = len(points)
    total_time = 0

    for i in range(number_of_points):
        if i == 0 or i == number_of_points - 1:
            continue
        
        previous = i - 1
        current = i
        following = i + 1

        previous_point = points[previous]
        current_point = points[current]
        following_point = points[following]

        current_speed = speed_profile[current]
        following_speed = speed_profile[following]

        stretch_distance = get_stretch_distance(current_point, following_point)
        acceleration = (following_speed**2 - current_speed**2) / (2*stretch_distance)

        stretch_time = (following_speed - current_speed) / acceleration if acceleration != 0 else stretch_distance / current_speed
        penalty = calculate_angle_penalty(previous_point, current_point, following_point)
        
        total_time += stretch_time + penalty
        # print(f"total: {total_time}")

        if stretch_time < 0:
            print(following_speed)
    
    return total_time

def lapTime(waypoints: list, track_start: int, track_end: int, max_speed: float) -> list:
    path = generate_path(waypoints, track_start, track_end)
    speed_profile = generate_speed_profile(path, max_speed)
    time = calculate_time(path, speed_profile)

    return [time, path, speed_profile]

def generations_to_plot(amount: int, step: int) -> list:
    return np.arange(1, amount + 1) * step

def run():
    start_time = time.time()
    F = 0.5
    CR = 0.85
    max_gen = 100
    pop_size = 1400
    track_start = 0
    track_end = 70
    gens_to_plot = generations_to_plot(10, 10)

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
        track_start=track_start,
        track_end=track_end,
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

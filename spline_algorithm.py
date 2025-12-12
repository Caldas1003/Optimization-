from datetime import datetime
from path import PATH
import matplotlib.pyplot as plt
import time
import numpy as np
import warnings
import os

from PistaComAtrito import TRACK
from diferential_evolution import customDifferentialEvolution


def spline_equation(t: float, a: float, b: float, c: float) -> float:
    return a*(t**2) + b*t + c

def first_derivative(t: float, a: float, b: float, c: float) -> float:
    return 2*a*t + b

def calculate_coeficients(
    spline_at_t0: float,
    spline_at_t1: float,
    spline_first_derivative_at_t0: float,
) -> dict:
    c = spline_at_t0
    b = spline_first_derivative_at_t0
    a = spline_at_t1 - b - c

    return { "a": a, "b": b, "c": c }

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

def build_splines(waypoints: list, checkpoints: list) -> list:
    splines = list()

    for i in range(len(checkpoints) - 1):							
        can_go_straight = i == 0
        first_derivative_at_t0 = [0, 0]

        X_k, Y_k, Z_k = TRACK.CHECKPOINTS[checkpoints[i]][waypoints[i]]
        X_k_plus_1, Y_k_plus_1, Z_k_plus_1 = TRACK.CHECKPOINTS[checkpoints[i + 1]][waypoints[i + 1]]

        if i > 0:
            X_k_minus_1, Y_k_minus_1, Z_k_minus_1 = TRACK.CHECKPOINTS[checkpoints[i - 1]][waypoints[i - 1]]
            full_turn_radius = get_turn_radius([X_k_minus_1, Y_k_minus_1], [X_k, Y_k], [X_k_plus_1, Y_k_plus_1])
            can_go_straight = full_turn_radius >= 33 # allows max speed (22.22 m/s)

            previous_spline = splines[i - 1]
            X_spline, Y_spline = previous_spline
            # t = 1 because the first derivative of the current segment at t=0 should be equal to the first derivative of the previous segment at t=1, same goes for seconds
            first_derivative_at_t0 = [
                    first_derivative(1, X_spline["a"], X_spline["b"], X_spline["c"]),
                    first_derivative(1, Y_spline["a"], Y_spline["b"], Y_spline["c"]),
            ]

        
        if can_go_straight:
            X_coefs = { "a": 0, "b": (X_k_plus_1 - X_k), "c": X_k }
            Y_coefs = { "a": 0, "b": (Y_k_plus_1 - Y_k), "c": Y_k }
        else:
            X_coefs = calculate_coeficients(X_k, X_k_plus_1, first_derivative_at_t0[0])
            Y_coefs = calculate_coeficients(Y_k, Y_k_plus_1, first_derivative_at_t0[1])
            
        splines.append([X_coefs, Y_coefs])
    
    return splines

def generate_path(waypoints: list, checkpoints: list, ds: float = 3) -> list:
    splines = build_splines(waypoints, checkpoints)
    path = []

    for i in range(len(checkpoints) - 1):
        X_spline, Y_spline = splines[i]
        x0 = spline_equation(0, X_spline["a"], X_spline["b"], X_spline["c"])
        y0 = spline_equation(0, Y_spline["a"], Y_spline["b"], Y_spline["c"])
        x1 = spline_equation(1, X_spline["a"], X_spline["b"], X_spline["c"])
        y1 = spline_equation(1, Y_spline["a"], Y_spline["b"], Y_spline["c"])

        distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        steps = np.round(distance / ds, 0)
        step = 1 / steps

        if i == 0:
            path.append([x0, y0])
            
        for t in np.arange(step, 1.1, step):
            x = spline_equation(t, X_spline["a"], X_spline["b"], X_spline["c"])
            y = spline_equation(t, Y_spline["a"], Y_spline["b"], Y_spline["c"])
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

def calculate_parable_half(x_max: float, y_max: float) -> dict:
    c3 = y_max
    c2 = 0
    c1 = - y_max / (x_max**2)

    return {"a": c1, "b": c2, "c": c3}

def generate_gg_diagram(pure_acceleration: float, pure_breaking: float, pure_cornering: float) -> dict:
    """returns the functions representing the cars gg diagram
    
    Keyword arguments:
    pure_acceleration -- maximum positive value for pure acceleration (m/s²)
    pure_breaking -- maximum negative value for pure breaking (m/s²)
    pure_cornering -- maximum absolute value for pure cornering (m/s²)
    Return: dict with the functions for acceleration and breaking, along with the pure cornering value, keys: 'acceleration', 'breaking', 'cornering'. 
    """
    acceleration = calculate_parable_half(pure_cornering, pure_acceleration)
    breaking = calculate_parable_half(pure_cornering, pure_breaking)

    return {"acceleration": acceleration, "breaking": breaking, "cornering": pure_cornering}

def generate_speed_profile(points:list, gg_diagram: dict, max_speed: float) -> list:
    speed_profile = []
    number_of_points = len(points)
    max_speed = np.float64(max_speed)

    max_lateral_acc = gg_diagram['cornering']

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

        if turn_radius > 0:
            max_turn_speed = np.sqrt(max_lateral_acc * turn_radius)
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
            max_braking = gg_diagram['breaking']['c'] # should be negative
            turn_radius = get_turn_radius(previous_point, current_point, following_point)
            if turn_radius > 0:
                lateral_acc = speed_profile[previous]**2 / turn_radius
                max_braking = gg_diagram['breaking']['a']*(lateral_acc**2) + gg_diagram['breaking']['b']*lateral_acc + gg_diagram['breaking']['c'] # should be negative
            stretch_distance = get_stretch_distance(previous_point, current_point)
            speed = np.sqrt(speed_profile[previous]**2 + 2*stretch_distance*max_braking)
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
            max_accelaration = gg_diagram['acceleration']['c']
            turn_radius = get_turn_radius(previous_point, current_point, following_point)
            if turn_radius > 0:
                lateral_acc = speed_profile[previous]**2 / turn_radius
                max_accelaration = gg_diagram['acceleration']['a']*(lateral_acc**2) + gg_diagram['acceleration']['b']*lateral_acc + gg_diagram['acceleration']['c']
            stretch_distance = get_stretch_distance(previous_point, current_point)
            speed = np.sqrt(speed_profile[previous]**2 + 2*stretch_distance*max_accelaration)
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

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # catch all warnings
        value = scalar_product / (magnitude_A * magnitude_B)
        value = 1 if value > 1 else value
        value = -1 if value < -1 else value
        angle_in_rad = np.arccos(value)

        # Check if any warnings were triggered
        if w:
            for warn in w:
                print(f"Caught warning: {warn.message}")
                print(f"value: {value}")
                print(value > 1)
    
    if np.degrees(angle_in_rad) > 180:
        print("Ângulo maior que 180 graus, o que não é esperado.")

    return 180 - np.degrees(angle_in_rad)

def calculate_angle_penalty(P1, P2, P3):
    """
    Penaliza curvas com ângulos pequenos.
    Quanto menor o ângulo, maior a penalização.
    """
    angle = calculate_angle(P1, P2, P3)
    penalty = (20/(90**2))*((180 - angle)**2)

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

        # previous_point = points[previous]
        current_point = points[current]
        following_point = points[following]

        current_speed = speed_profile[current]
        following_speed = speed_profile[following]

        stretch_distance = get_stretch_distance(current_point, following_point)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            acceleration = (following_speed**2 - current_speed**2) / (2*stretch_distance)
            # if w:
            #     print(f"Warning: {w[0].message}")
            #     print(f"current_point: {current_point}")
            #     print(f"following_point: {following_point}")

        stretch_time = (following_speed - current_speed) / acceleration if acceleration != 0 else stretch_distance / current_speed
        # penalty = calculate_angle_penalty(previous_point, current_point, following_point)

        total_time += stretch_time
        # print(f"total: {total_time}")

        if stretch_time < 0:
            print(following_speed)
    
    return total_time

def lapTime(max_speed: float, pure_acceleration: float, pure_breaking: float, pure_cornering: float) -> list:
    path = np.array(PATH)
    gg_diagram = generate_gg_diagram(pure_acceleration=pure_acceleration, pure_breaking=pure_breaking, pure_cornering=pure_cornering)
    speed_profile = generate_speed_profile(path, gg_diagram=gg_diagram, max_speed=max_speed)
    time = calculate_time(path, speed_profile)

    return [time, speed_profile]

def generations_to_plot(amount: int, step: int) -> list:
    return np.arange(1, amount + 1) * step

def run():
    start_time = time.time()
    F = 0.5
    CR = 0.3
    max_gen = 200
    pop_size = 100
    track_start = 8
    track_end = 35
    gens_to_plot = generations_to_plot(20, 10)
    bounds = [
        [0.4, 0.6], # distribution
        [2.5e5, 7.5e5], # Kds
        [2.5e5, 7.5e5], # Kts
        [1e4, 3e4], # Kpds
        [1e4, 3e4], # Kpts
        [1e5, 5e5], # Kf
        [0.8e5, 2.4e5], # Kt
        [1e4, 5e4], # Cds
        [2.5e3, 7.5e3], # Cts
        [0.9, 1.1], # W
        [1, 1.2], # Lt
        [1.2, 1.6], # Ld
        [0.278, 22.22], # max speed
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d %H%M")
    folder = f"{timestamp} - speed profile gen"
    os.makedirs(folder, exist_ok=True)

    best_params, best_fitness, standard_deviation, data = customDifferentialEvolution(
        lapTime,
        np.array(bounds),
        max_generations=max_gen,
        F=F,
        CR=CR,
        pop_size=pop_size,
        gensToPlot=gens_to_plot,
        folder=folder
    )

    time_elapsed = time.time() - start_time
    hours, seconds_remain = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(seconds_remain, 60)

    print(f"Melhores waypoints: {best_params[0]}")
    print(f"Speed profile: {best_params[1]}")
    print(f"Velocidade mínima: {best_params[1].min()}")
    print(f"Velocidade máxima: {best_params[1].max()}")
    print(f"Velocidade média: {best_params[1].mean()}")
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

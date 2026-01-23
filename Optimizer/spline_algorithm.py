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
    c1 = - y_max / ((x_max + 1e-6)**2)

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
        if i == 0:
            continue
        
        # current é o ponto onde o carro chega, previous é de onde veio
        # Ajuste conforme a lógica do seu loop (no seu código original parecia usar i, i+1)
        # Analisando seu código original: você iterava de 0 a len, ignorando o último e primeiro
        # Mas usava current=i e following=i+1. Vamos manter essa lógica.
        
        if i == number_of_points - 1:
             continue

        current = i
        following = i + 1

        current_point = points[current]
        following_point = points[following]

        current_speed = speed_profile[current]
        following_speed = speed_profile[following]

        stretch_distance = get_stretch_distance(current_point, following_point)

        # --- CORREÇÃO AQUI ---
        # Cálculo usando Velocidade Média (muito mais estável)
        # t = distancia / velocidade_media
        avg_speed = (current_speed + following_speed) / 2.0

        # Proteção contra divisão por zero (caso o otimizador tente parar o carro)
        if avg_speed > 1e-3: 
            stretch_time = stretch_distance / avg_speed
        else:
            # Se a velocidade for quase zero, aplicamos uma penalidade gigante de tempo
            # para que o algoritmo genético descarte essa solução.
            # print(f"Aviso: Velocidade muito baixa no índice {i}. Aplicando penalidade.")
            stretch_time = 1e6 

        total_time += stretch_time

        if stretch_time < 0:
            print(f"Erro: Tempo negativo detectado: {stretch_time}")

    return total_time

def physical_penalties(params, load_transfer=None):
    """
    Penalidades físicas e relacionais para FSAE SEM aerodinâmica.
    Retorna:
        total, penalties, interactions, load_penalties
    """

    # ==============================================================
    # Funções auxiliares
    # ==============================================================

    def soft_penalty(x, xmin=None, xmax=None, scale=1.0):
        p = 0.0
        if xmin is not None and x < xmin:
            p += ((xmin - x) / xmin)**2 * scale
        if xmax is not None and x > xmax:
            p += ((x - xmax) / xmax)**2 * scale
        return p

    def ratio_penalty(r, rmin, rmax, scale=1.0):
        if r < rmin:
            return ((rmin - r) / rmin)**2 * scale
        if r > rmax:
            return ((r - rmax) / rmax)**2 * scale
        return 0.0

    # ==============================================================
    # Extração dos parâmetros
    # ==============================================================

    dist = params["distribution"]     # fração 

    Kf_susp = params["Kds"]            # suspensão dianteira
    Kr_susp = params["Kts"]            # suspensão traseira

    Cf = params["Cds"]                 # amortecimento dianteiro
    Cr = params["Cts"]                 # amortecimento traseiro

    Kp = 0.5 * (params["Kpds"] + params["Kpts"])  # pneus

    Kf_chassi = params["Kf"]           # flexão
    Kt_chassi = params["Kt"]           # torção

    W  = params["W"]                   # bitola
    Lt = params["Lt"]                  # entre-eixos traseiro
    Ld = params["Ld"]                  # entre-eixos dianteiro

    penalties = {}
    interactions = {}
    load_penalties = {}

    # ==============================================================
    # 1️⃣ Penalidades individuais
    # ==============================================================
    
    ### RESOLVIDAS PELOS BOUNDS DO OTIMIZADOR ###

    # penalties["distribution"] = soft_penalty(dist, 0.45, 0.55, scale=4)

    # penalties["spring_front"] = soft_penalty(Kf_susp, 2.5e5, 7.5e5, scale=2)
    # penalties["spring_rear"]  = soft_penalty(Kr_susp, 2.5e5, 7.5e5, scale=2)

    # penalties["damping_front"] = soft_penalty(Cf, 1e4, 5e4, scale=1.5)
    # penalties["damping_rear"]  = soft_penalty(Cr, 2.5e3, 7.5e3, scale=1.5)

    # penalties["tire_stiffness"] = soft_penalty(Kp, 1e4, 3e4, scale=1.5)

    # penalties["chassis_flex"]    = soft_penalty(Kf_chassi, 1e6, 3e7, scale=3)
    # penalties["chassis_torsion"] = soft_penalty(Kt_chassi, 8e2, 2.4e7, scale=3)

    # penalties["track_width"]      = soft_penalty(W, 0.9, 1.1, scale=1)
    # penalties["wheelbase_front"]  = soft_penalty(Ld, 1.2, 1.6, scale=1)
    # penalties["wheelbase_rear"]   = soft_penalty(Lt, 1.0, 1.2, scale=1)

    # ==============================================================
    # 2️⃣ Penalidades RELACIONAIS (núcleo do modelo)
    # ==============================================================

    # Distribuição de massa × rigidez de suspensão
    interactions["mass_vs_spring_balance"] = ratio_penalty(
        (Kf_susp / Kr_susp),
        (dist / (1 - dist)) * 0.8,
        (dist / (1 - dist)) * 1.25,
        scale=6
    )

    # Gradiente de subesterço (frente levemente mais rígida)
    interactions["understeer_bias"] = ratio_penalty(
        Kf_susp / Kr_susp,
        0.95, 1.30,
        scale=5
    )

    # Frequência natural frente × traseira (adimensional)
    interactions["freq_balance"] = ratio_penalty(
        np.sqrt(Kf_susp / Kr_susp),
        0.9, 1.15,
        scale=5
    )

    # Suspensão × pneu
    interactions["spring_vs_tire_front"] = ratio_penalty(
        Kf_susp / Kp,
        0.6, 2,
        scale=3
    )

    interactions["spring_vs_tire_rear"] = ratio_penalty(
        Kr_susp / Kp,
        0.6, 2,
        scale=3
    )

    # Chassi × suspensão (modo estrutural)
    interactions["chassis_vs_susp"] = ratio_penalty(
        Kf_chassi / (Kf_susp + Kr_susp),
        10, 500,
        scale=2
    )

    # Torção × rigidez ao rolamento
    roll_susp = (Kf_susp + Kr_susp) * (W**2) / 4

    interactions["torsion_vs_roll"] = ratio_penalty(
        Kt_chassi / roll_susp,
        5, 100,
        scale=6
    )

    # Bitola × rigidez total (transferência de carga)
    interactions["track_vs_stiffness"] = ratio_penalty(
        (Kf_susp + Kr_susp) * W,
        2e5, 1.2e6,
        scale=4
    )

    # Geometria longitudinal × distribuição de massa
    interactions["wheelbase_vs_distribution"] = ratio_penalty(
        (Ld / (Ld + Lt)) / dist,
        0.85, 1.15,
        scale=5
    )

    # Amortecimento × mola (escala física)

    # Estimativa de massa suspensa por canto (ajuste conforme seu carro)
    m_car_total = 300.0 # kg (com piloto)
    m_unsprung = 40.0   # kg (total 4 rodas)
    m_suspended = m_car_total - m_unsprung

    m_front_corner = (m_suspended * dist) / 2
    m_rear_corner  = (m_suspended * (1 - dist)) / 2

    zeta_front = Cf / (2 * np.sqrt(m_front_corner * Kf_susp))
    zeta_rear  = Cr / (2 * np.sqrt(m_rear_corner * Kr_susp))

    interactions["damping_ratio_front"] = ratio_penalty(zeta_front, 0.5, 0.9, scale=10)
    interactions["damping_ratio_rear"]  = ratio_penalty(zeta_rear, 0.5, 0.9, scale=10)

    # ==============================================================
    # 3️⃣ Penalidades por transferência de carga (opcional)
    # ==============================================================

    # if load_transfer is not None:
    #     load_penalties["lat_front"] = soft_penalty(
    #         load_transfer["lat_front"], 0.0, 0.65, scale=2
    #     )
    #     load_penalties["lat_rear"] = soft_penalty(
    #         load_transfer["lat_rear"], 0.0, 0.65, scale=2
    #     )
    #     load_penalties["long_front"] = soft_penalty(
    #         load_transfer["long_front"], 0.0, 0.7, scale=2
    #     )
    #     load_penalties["long_rear"] = soft_penalty(
    #         load_transfer["long_rear"], 0.0, 0.7, scale=2
    #     )

    # ==============================================================
    # 4️⃣ Penalidade total
    # ==============================================================

# Soma ponderada (pesos que você definiu)
    total_individual = 1.0 * sum(penalties.values())
    total_interactions = 2.0 * sum(interactions.values())
    total_load = 1.5 * sum(load_penalties.values())

    penalty_raw = total_individual + total_interactions + total_load

    
    
    penalty_normalized = penalty_raw / (1.0 + penalty_raw)

    return penalty_normalized, penalties, interactions, load_penalties  

def print_penalty_report(penalties, interactions, load_penalties, threshold=1e-4):
    """
    Imprime um relatório detalhado das penalidades ativas.
    threshold: valor mínimo para considerar a penalidade relevante (filtra ruídos).
    """
    print("\n" + "="*60)
    print(f"{'VIOLAÇÃO':<35} | {'PENALIDADE':<10} | {'STATUS'}")
    print("-" * 60)

    # Junta tudo em um único dicionário para análise
    all_data = {**penalties, **interactions, **load_penalties}
    
    # Ordena da maior penalidade para a menor
    sorted_penalties = sorted(all_data.items(), key=lambda item: item[1], reverse=True)
    
    active_count = 0
    for name, value in sorted_penalties:
        if value > threshold:
            active_count += 1
            # Formatação visual para gravidade
            severity = "CRÍTICO" if value > 10 else "ALTO" if value > 1 else "Médio" if value > 0.1 else "Baixo"
            print(f"{name:<35} | {value:10.4f} | {severity}")

    if active_count == 0:
        print(f"{'Nenhuma penalidade significativa encontrada!':^60}")
    
    print("="*60 + "\n")

def lapTime(max_speed: float, pure_acceleration: float, pure_breaking: float, pure_cornering: float, individual, debug=False) -> list:
    path = np.array(PATH)
    gg_diagram = generate_gg_diagram(pure_acceleration=pure_acceleration, pure_breaking=pure_breaking, pure_cornering=pure_cornering)
    speed_profile = generate_speed_profile(path, gg_diagram=gg_diagram, max_speed=max_speed)
    (   distribution,    Kds, Kts,    Kpds, Kpts,    Kf, Kt,    Cds, Cts,    W, Lt, Ld) = individual
    params = {
    "distribution": distribution,
    "Kds": Kds,
    "Kts": Kts,
    "Kpds": Kpds,
    "Kpts": Kpts,
    "Kf": Kf,
    "Kt": Kt,
    "Cds": Cds,
    "Cts": Cts,
    "W": W,
    "Lt": Lt,
    "Ld": Ld,}

    #load_transfer = {
    #"lat_front": 0.5,
    #"lat_rear": 0.5,
    #"long_front": 0.5,
    #"long_rear": 0.5,}

    penalty_norm, pens, inters, loads = physical_penalties(params)
    
    # Lógica de Fitness
    time = calculate_time(path, speed_profile)
    
    # Se o debug estiver ligado OU se a penalidade for muito alta (o carro está "quebrado")
    if debug or penalty_norm > 0.5: 
        print(f"DEBUG: Tempo calculado: {time:.2f}s | Penalidade Norm: {penalty_norm:.2%}")
        # Chama nosso relatório
        print_penalty_report(pens, inters, loads)

    # Aplica a penalidade no tempo (como estava no seu comentário)
    # Se penalty_norm for 0.5 (50%), o tempo aumenta em 50%.
    fitness = time * (1.0 + penalty_norm) 

    return fitness, speed_profile

def generations_to_plot(amount: int, step: int) -> list:
    return np.arange(1, amount + 1) * step

def save_results(
    best_params,
    best_fitness,
    standard_deviation,
    max_gen,
    pop_size,
    F,
    CR,
    elapsed_time,      # ⬅️ não usar "time" (conflito com módulo)
    time_stamp,
    folder_path
):
    """
    Salva os resultados da otimização em arquivo texto.
    """

    # -----------------------------
    # Organização dos dados
    # -----------------------------
    hours, minutes, seconds = elapsed_time

    parameters, speed_profile = best_params  # unpack correto

    filename = f"{time_stamp}_gen={max_gen}_pop={pop_size}.txt"
    file_path = os.path.join(folder_path, filename)

    # -----------------------------
    # Conteúdo do arquivo
    # -----------------------------
    content = []
    content.append("==== RESULTADOS DA OTIMIZAÇÃO ====\n")
    content.append(f"F = {F}\n")
    content.append(f"CR = {CR}\n")
    content.append(f"Gerações máximas = {max_gen}\n")
    content.append(f"Tamanho da população = {pop_size}\n\n")

    content.append("---- MELHORES PARÂMETROS ----\n")

    param_names = [
        "Distribuição de peso",
        "Kds (mola dianteira)",
        "Kts (mola traseira)",
        "Kpds (pneu dianteiro)",
        "Kpts (pneu traseiro)",
        "Kf (flexão do chassi)",
        "Kt (torção do chassi)",
        "Cds (amortecimento dianteiro)",
        "Cts (amortecimento traseiro)",
        "W (bitola)",
        "Lt (entre-eixos traseiro)",
        "Ld (entre-eixos dianteiro)"
    ]

    for name, value in zip(param_names, parameters):
        content.append(f"{name}: {value:.6e}\n")

    content.append("\n---- RESULTADOS ----\n")
    content.append(f"Menor tempo total encontrado: {best_fitness:.6f} s\n")
    content.append(f"Desvio padrão final: {standard_deviation:.6f}\n")
    content.append(
        f"Tempo total de execução: "
        f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}\n"
    )

    content.append("\n---- PERFIL DE VELOCIDADE ----\n")
    content.append(
        f"Velocidade mínima: {np.min(speed_profile):.3f} m/s\n"
        f"Velocidade máxima: {np.max(speed_profile):.3f} m/s\n"
        f"Velocidade média: {np.mean(speed_profile):.3f} m/s\n"
    )

    # -----------------------------
    # Escrita do arquivo
    # -----------------------------
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(content)

    print(f"Resultados salvos em: {file_path}")

def run():
    start_time = time.time()
    F = 0.6
    CR = 0.3
    max_gen = 10
    pop_size = 5
    track_start = 0
    track_end = 200
    gens_to_plot = generations_to_plot(20, 10)
    bounds = [
        [0.4, 0.6],         # distribution
        [1.5e4, 6.5e4],     # Rigidez da mola dianteira
        [1.5e4, 6.5e4],     # Kts
        [9.0e4, 1.5e5],     # Kpds
        [9.0e4, 1.5e5],     # Kpts
        [1e5, 5e7],         # Kf
        [0.8e5, 2.4e7],     # Kt
        [1.0e3, 5.0e3],     # Cds
        [1.0e3, 5.0e3],     # Cts
        [0.9, 1.1],         # W
        [1, 1.2],           # Lt
        [1.2, 1.6],         # Ld
        [0.278, 22.22],     # max speed
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
        track_start=0,
        track_end=200,
        gensToPlot=gens_to_plot,
        folder=folder
    )

    time_elapsed = time.time() - start_time
    hours, seconds_remain = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(seconds_remain, 60)

    save_results(
        best_params=best_params,
        best_fitness=best_fitness,
        standard_deviation=standard_deviation,
        max_gen=max_gen,
        pop_size=pop_size,
        F=F,
        CR=CR,
        elapsed_time=(hours, minutes, seconds),
        time_stamp=timestamp,
        folder_path=folder
    )
    print(f"Melhores parametros: {best_params[0]}")
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

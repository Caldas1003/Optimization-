from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from scipy.spatial import cKDTree
import time
from shapely.geometry import Polygon
import numpy as np
import warnings
import os

from Pista import TRACK
from diferential_evolution import customDifferentialEvolution

"""
    PENALIDADES ANTIGAS :
    COMPUTE_SMOOTHNESS_PENALTY SOZINHA TEM O MELHOR RESULTADO
"""
def compute_smoothness_penalty(waypoints,weight = 0.15):
    penalty = 0.0
    n = len(waypoints)
    
    for i in range(1, n - 1):
        # Pontos adjacentes (3 pontos)
        p0 = TRACK.CHECKPOINTS[i - 1][waypoints[i - 1]]
        p1 = TRACK.CHECKPOINTS[i][waypoints[i]]
        p2 = TRACK.CHECKPOINTS[i + 1][waypoints[i + 1]]
        
        # Cálculo básico de curvatura (3 pontos)
        mid_point = (p0 + p2) / 2
        curvature_3pt = np.linalg.norm(p1 - mid_point)
        
        # Cálculo estendido com 5 pontos (quando aplicável)
        if i >= 2 and i < n - 2:
            p_prev = TRACK.CHECKPOINTS[i - 2][waypoints[i - 2]]  # Ponto anterior distante
            p_next = TRACK.CHECKPOINTS[i + 2][waypoints[i + 2]]  # Próximo ponto distante
            
            # Curvatura usando 5 pontos (ajuste quadrático implícito)
            curvature_5pt = np.linalg.norm(
                (p_prev + 4*p0 + 6*p1 + 4*p2 + p_next) / 16 - p1
            )
            # Combina as duas medidas
            penalty += 0.4 * curvature_3pt + 0.6 * curvature_5pt
        else:
            penalty += curvature_3pt
    
    return penalty * weight

def yaw_or_latacc_penalty(path, speed_profile,max_lateral_g=1.5,max_longitudinal_g=1.2, w_excess=2.0,w_smooth=1.0):
    g = 9.81
    ay_limit = max_lateral_g * g
    ax_limit = max_longitudinal_g * g
    penalty = 0.0

    if len(path) < 3:
        return 0.0

    # calcula curvaturas e distâncias
    curvatures = []
    ds_list = []
    for i in range(1, len(path)-1):
        R = get_turn_radius(path[i-1], path[i], path[i+1])
        ds = get_stretch_distance(path[i], path[i+1])
        ds_list.append(max(ds, 1e-6))
        curvatures.append(0.0 if R == 0 else 1.0 / R)

    # penalidades por ponto (usar a_x estimado por diferenças de velocidade)
    for i in range(1, len(path)-1):
        v = speed_profile[i]
        kappa = curvatures[i-1]
        ds = ds_list[i-1]
        ay = (v**2) * abs(kappa)      # v^2 * kappa
        # excesso lateral
        if ay > ay_limit:
            penalty += w_excess * ((ay - ay_limit)**2) * ds

        # estimativa simples de a_x (central diff on speeds)
        # index shift: speed_profile length == path length
        ax = 0.0
        if 0 < i < len(path)-1:
            dt = ds / max(v, 1e-6)
            ax = (speed_profile[i+1] - speed_profile[i-1]) / max(2*dt, 1e-6)
            # elipse de atrito: penaliza combinação ax/ax_lim e ay/ay_lim > 1
            lhs = (ax/ax_limit)**2 + (ay/ay_limit)**2
            if lhs > 1.0:
                penalty += w_excess * ((lhs - 1.0)**2) * ds

    # suavidade: penaliza variação de curvatura (Δκ^2), ponderado por ds média
    for i in range(1, len(curvatures)):
        dkap = curvatures[i] - curvatures[i-1]
        ds_avg = 0.5*(ds_list[i] + ds_list[i-1]) if i < len(ds_list) else ds_list[i-1]
        penalty += w_smooth * (dkap**2) * ds_avg

    return penalty

def compute_lateral_acc(path, speed_profile, max_lateral_g=1.5, w_excess=2.0, w_smooth=1.0):
    g = 9.81
    max_lateral_acc = max_lateral_g * g
    penalty = 0.0

    if len(path) < 3:
        return 0.0

    curvatures = []
    for i in range(1, len(path) - 1):
        R = get_turn_radius(path[i-1], path[i], path[i+1])
        if R <= 0:
            curvatures.append(0.0)
            continue
        
        v = speed_profile[i]
        ay = (v**2) / R
        ds = get_stretch_distance(path[i], path[i+1])

        # Penaliza excesso
        if ay > max_lateral_acc:
            penalty += w_excess * ((ay - max_lateral_acc)**2) * ds

        curvatures.append(1.0 / R)

    # Penalidade de suavidade baseada em variação de curvatura
    for i in range(1, len(curvatures)):
        dkap = curvatures[i] - curvatures[i-1]
        penalty += w_smooth * (dkap**2)

    return penalty

def compute_yaw_rate_penalty_path(path, speed_profile, max_yaw_rate_rad_per_sec, weight=0.05):
    penalty = 0.0

    if len(path) < 3:
        return 0.0

    for i in range(len(path) - 2):
        P1, P2, P3 = path[i], path[i+1], path[i+2]

        vec_segment1 = P2 - P1
        vec_segment2 = P3 - P2

        angle_segment1 = np.arctan2(vec_segment1[1], vec_segment1[0])
        angle_segment2 = np.arctan2(vec_segment2[1], vec_segment2[0])

        delta_angle = (angle_segment2 - angle_segment1 + np.pi) % (2*np.pi) - np.pi

        avg_speed_segment = (speed_profile[i] + speed_profile[i+1]) / 2.0
        segment_distance = get_stretch_distance(P1, P2)

        if avg_speed_segment > 0 and segment_distance > 0:
            time_in_segment = segment_distance / avg_speed_segment
            yaw_rate = abs(delta_angle / time_in_segment)

            if yaw_rate > max_yaw_rate_rad_per_sec:
                penalty += weight * (yaw_rate - max_yaw_rate_rad_per_sec)**2
        else:
            penalty += 10.0 * (yaw_rate - max_yaw_rate_rad_per_sec)**2

    return penalty

def smoothness_penalty_from_path(path, w_k=0.05, w_dk=0.02):
    kappa, ds = curvature_series(path)
    if len(kappa) < 2:
        return 0.0
    P_curv = np.sum((kappa**2) * ds[:len(kappa)])
    dkap = np.diff(kappa)
    ds_mid = 0.5*(ds[:-1]+ds[1:])
    P_dk = np.sum((dkap**2)/np.maximum(ds_mid, 1e-6))
    return w_k*P_curv + w_dk*P_dk
"""
    FIM DAS PENALIDADES ANTIGAS
"""

"""
    PENALIDADES NOVAS COM O USO DE VETOR DIREÇÃO(HEADINGS)
"""

def compute_headings(path: np.ndarray):
    """
    Retorna headings (vetores normalizados) e distâncias ds entre pontos consecutivos.
    """
    deltas = path[1:] - path[:-1]
    ds = np.linalg.norm(deltas, axis=1) + 1e-9
    headings = deltas / ds[:, None]
    return headings, ds

def penalty_smoothness(headings, ds_list, w=2.5,tolerance=0.1):
    penalty = 0.0
    curvatures = []
    for i in range(1, len(headings)):
        delta_h = headings[i] - headings[i-1]
        kappa = np.linalg.norm(delta_h) / ds_list[i-1]
        curvatures.append(kappa)

    if len(curvatures) > 1:
        dkap = np.diff(curvatures)
        
        # Cria uma máscara booleana para encontrar os valores que excedem a tolerância
        exceeds_tolerance = np.abs(dkap) > tolerance
        penalized_dkap = np.where(exceeds_tolerance, np.abs(dkap) - tolerance, 0)
        penalty += w * np.sum(penalized_dkap**2)

    return penalty

def penalty_lateral_longitudinal(path, speed_profile,headings, ds_list, max_lateral_g=1.3, max_longitudinal_g=1.5, w=2.0):
    g = 9.81
    ay_limit = max_lateral_g * g
    ax_limit = max_longitudinal_g * g
    penalty = 0.0
    n = len(path)

    for i in range(1, n-1):
        v = speed_profile[i]
        ds = ds_list[i-1]

        # curvatura ≈ variação de heading / ds
        delta_h = headings[i] - headings[i-1]
        kappa = np.linalg.norm(delta_h) / ds

        ay = v**2 * kappa
        if ay > ay_limit:
            penalty += w * ((ay - ay_limit)**2) * ds

        if 0 < i < n-2:
            dt = ds / max(v, 1e-6)
            ax = (speed_profile[i+1] - speed_profile[i-1]) / max(2*dt, 1e-6)
            lhs = (ax/ax_limit)**2 + (ay/ay_limit)**2
            if lhs > 1.0:
                penalty += w * ((lhs - 1.0)**2) * ds

    return penalty

def penalty_yaw_rate(speed_profile, headings, ds_list, max_yaw_rate=0.8, w=1.5):
    penalty = 0.0

    for i in range(1, len(headings)):
        h1 = headings[i-1]  # Vetor 2D [x, y]
        h2 = headings[i]    # Vetor 2D [x, y]
        # Calcula o produto escalar e a magnitude do produto vetorial (2D)
        dot = h1[0]*h2[0] + h1[1]*h2[1]
        cross = h1[0]*h2[1] - h1[1]*h2[0]
        # Ângulo entre os vetores
        delta_angle = np.arctan2(cross, dot)
        # Velocidade e distância
        v = speed_profile[i]
        ds = ds_list[i-1]
        # Taxa de guinada (yaw rate)
        yaw_rate = abs(delta_angle) / (ds / max(v, 1e-6))
        if yaw_rate > max_yaw_rate:
            penalty += w * (yaw_rate - max_yaw_rate)**2

    return penalty


"""
    FIM DAS PENALIDADES NOVAS
"""

"""
    IMPLEMENTAÇÃO DO CARRO COMO UM MODELO 2D SIMPLES
"""


def get_car_polygon(x, y, heading, L=2.0, W=1.2):
    """
    Retorna os 4 vértices do carro (retângulo) baseado na posição e orientação.
    (x, y) = centro do carro
    heading = ângulo em radianos
    L = comprimento do carro
    W = largura do carro
    """
    half_L = L / 2.0
    half_W = W / 2.0

    # pontos em coordenadas locais
    local_points = np.array([
        [ half_L,  half_W],
        [ half_L, -half_W],
        [-half_L, -half_W],
        [-half_L,  half_W]
    ])

    # rotação
    R = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading),  np.cos(heading)]
    ])

    # aplicar rotação e translação
    global_points = (R @ local_points.T).T + np.array([x, y])
    return global_points

def car_inside_track_fast(x, y, heading, i, track, L, W):
    idx = np.argmin((track.X_suave - x) ** 2 + (track.Y_suave - y) ** 2)

    n = len(track.X_suave)
    j = (i + 1) % n
    vdir = np.array([track.X_suave[j] - track.X_suave[idx],
                     track.Y_suave[j] - track.Y_suave[idx]])
    norm = np.linalg.norm(vdir)
    if norm == 0:
        return True
    vdir /= norm

    left = np.array([track.esquerda[idx][0], track.esquerda[idx][1]])
    right = np.array([track.direita[idx][0], track.direita[idx][1]])

    car_poly = get_car_polygon(x, y, heading, L, W)

    for px, py in car_poly:
        p = np.array([px, py])

        # promove para 3D
        cross_left = np.cross(np.append(vdir, 0), np.append(left - p, 0))[2]
        cross_right = np.cross(np.append(vdir, 0), np.append(p - right, 0))[2]

        if cross_left < 0 or cross_right < 0:
            return False

    return True

def collision_penalty(path, track, L=2.0, W=1.2, penalty_value=0.25,step=5):
    """
    Penaliza colisões do carro (modelo retangular 2D) ao longo do path,
    acumulando a penalidade para cada ponto de colisão.
    
    path -> lista/array de pontos [x,y]
    track -> objeto PistaComAtrito
    L, W -> dimensões do carro (m)
    penalty_value -> valor de penalidade por colisão
    """
    total_penalty = 0.0
    
    for i in range(0, len(path) - 1, step):
        x, y = path[i]
        x_next, y_next = path[i + 1]

        # heading do carro entre dois pontos do path
        heading = np.arctan2(y_next - y, x_next - x)

        # checar se carro está dentro da pista
        if not car_inside_track_fast(x, y, heading, i, track, L, W):
            total_penalty += penalty_value  # Adiciona a penalidade e continua o loop
            
    return total_penalty  # Retorna o total acumulado de penalidades

def car_inside_track_gradual(x, y, heading, idx, track, L, W):
    """
    Verifica se o carro está dentro da pista e retorna uma penalidade gradual.
    Retorna 0 se o carro estiver dentro, ou um valor > 0 se estiver fora.
    """
    n = len(track.X_suave)
    j = (idx + 1) % n
    vdir = np.array([track.X_suave[j] - track.X_suave[idx],
                     track.Y_suave[j] - track.Y_suave[idx]])
    norm = np.linalg.norm(vdir)

    if norm == 0:
        return 0  # Nenhum movimento, nenhuma penalidade
    vdir /= norm

    left = np.array([track.esquerda[idx][0], track.esquerda[idx][1]])
    right = np.array([track.direita[idx][0], track.direita[idx][1]])
    
    car_poly = get_car_polygon(x, y, heading, L, W)
    
    max_penalty = 0.0

    for px, py in car_poly:
        p = np.array([px, py])

        cross_left = np.cross(np.append(vdir, 0), np.append(left - p, 0))[2]
        cross_right = np.cross(np.append(vdir, 0), np.append(p - right, 0))[2]

        if cross_left < 0:
            max_penalty = max(max_penalty, abs(cross_left))
        if cross_right < 0:
            max_penalty = max(max_penalty, abs(cross_right))
            
    # O valor da penalidade pode ser escalonado aqui para ser mais significativo
    # Ex: return max_penalty * 100
    return max_penalty

# --- Nova Função de Penalidade (Otimização por Setores) ---
def collision_penalty_otimizada(path, track, L=2.0, W=1.2, step=5):
    """
    Penaliza colisões usando um modelo 2D, com otimização por setores e penalidade gradual.
    """
    total_penalty = 0.0
    last_idx = 0  # Começa buscando no início da pista
    search_range = 15 # Raio de busca para otimização do setor

    for i in range(0, len(path) - 1, step):
        x, y = path[i]
        x_next, y_next = path[i + 1]
        heading = np.arctan2(y_next - y, x_next - x)

        # 1. Otimização por setores: define o subconjunto de pontos para busca
        start_idx = max(0, last_idx - search_range)
        end_idx = min(len(track.X_suave), last_idx + search_range)
        
        subset_X = track.X_suave[start_idx:end_idx]
        subset_Y = track.Y_suave[start_idx:end_idx]
        
        # 2. Encontra o índice mais próximo dentro do setor
        if len(subset_X) > 0:
            local_idx = np.argmin((subset_X - x)**2 + (subset_Y - y)**2)
            current_idx = start_idx + local_idx
            last_idx = current_idx  # Atualiza o índice para a próxima iteração
        else:
            # Caso o carro esteja longe de qualquer setor, usa a busca completa
            current_idx = np.argmin((track.X_suave - x)**2 + (track.Y_suave - y)**2)
            last_idx = current_idx

        # 3. Aplica a penalidade gradual e acumula o valor
        total_penalty += car_inside_track_gradual(x, y, heading, current_idx, track, L, W)
            
    return total_penalty

"""
    FIM DA IMPLEMENTAÇÃO DO MODELO 2D
"""
def curvature_series(path):
    kappa, ds = [], []
    for i in range(1, len(path)-1):
        R = get_turn_radius(path[i-1], path[i], path[i+1])
        kappa.append(0.0 if R == 0 else 1.0/R)
        ds.append(get_stretch_distance(path[i], path[i+1]))
    return np.array(kappa), np.array(ds)

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

def calculate_time(points: list, speed_profile: list) -> float:
    if len(points) != len(speed_profile):
        raise ValueError("Number of points and speeds must be equal")
    
    number_of_points = len(points)
    total_time = 0
   
    for i in range(number_of_points):
        if i == 0 or i == number_of_points - 1:
            continue

        current = i
        following = i + 1

        current_point = points[current]
        following_point = points[following]
        current_speed = speed_profile[current]
        following_speed = speed_profile[following]

        stretch_distance = get_stretch_distance(current_point, following_point)
        acceleration = (following_speed**2 - current_speed**2) / (2*stretch_distance)

        stretch_time = (following_speed - current_speed) / acceleration if acceleration != 0 else stretch_distance / current_speed
        
        total_time += stretch_time
        # print(f"total: {total_time}")

        if stretch_time < 0:
            print(following_speed)
    
    return total_time

def lapTime(waypoints: list, track_start: int, track_end: int, max_speed: float, yaw_rate_limit: float = 0.8) -> list:
    path = generate_path(waypoints, track_start, track_end)
    speed_profile = generate_speed_profile(path, max_speed)
    time = calculate_time(path, speed_profile)
    

                ### PENALIDADES VELHAS
    #time += compute_smoothness_penalty(waypoints)
    #tempo_penalizado += smoothness_penalty_from_path(path)
    #tempo_penalizado += compute_lateral_acc(path,speed_profile)
    #tempo_penalizado += compute_yaw_rate_penalty_path(path, speed_profile, yaw_rate_limit)
    #tempo_penalizado += yaw_or_latacc_penalty(path, speed_profile)

                ## PENALIDADES NOVAS
    headings, ds = compute_headings(path)
    time += penalty_smoothness(headings, ds, w=0.8)
    #time += penalty_lateral_longitudinal(path, speed_profile, headings, ds, w=0.5)
    #time += penalty_yaw_rate(speed_profile, headings, ds, max_yaw_rate=yaw_rate_limit, w=0.2)
    #time += collision_penalty(path, TRACK)
    time +=collision_penalty_otimizada(path, TRACK)

    return [time, path, speed_profile]

def generations_to_plot(amount: int, step: int) -> list:
    return np.arange(1, amount + 1) * step

def save_results(best_params, best_fitness, standard_deviation, max_gen, pop_size,F,CR,time,time_stamp,folder_path):
    folder_path = folder_path
    hours,minutes,seconds = time
    filename = f"{time_stamp}_gen={max_gen}_pop={pop_size}.txt"
    file_path = os.path.join(folder_path, filename)
    
    content = (f"F = {F} , CR = {CR}\n"
            f"Melhores parâmetros encontrados: {best_params}\n"
            f"Menor tempo total encontrada: {best_fitness}\n"
            f"Desvio padrão: {standard_deviation}\n"
            f"Total time elapsed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\n")
    
   #print(content)
    
    with open(file_path, "w") as file:
        file.write(content)
    
    print(f"Resultados salvos em: {file_path}")

def animate_car_on_track_realistic(track, path, speed_profile, L=2.0, W=1.2, save_as=None):
    """
    Animação do carro percorrendo a pista com velocidade proporcional ao speed_profile.
    
    track          -> objeto da pista (classe PistaComAtrito)
    path           -> trajetória [(x,y), ...]
    speed_profile  -> velocidades em cada ponto
    L, W           -> dimensões do carro
    save_as        -> se fornecido, salva como arquivo .mp4 ou .gif
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bordas da pista
    X_esq, Y_esq, _ = zip(*track.esquerda)
    X_dir, Y_dir, _ = zip(*track.direita)
    ax.plot(X_esq, Y_esq, 'r--', linewidth=1)
    ax.plot(X_dir, Y_dir, 'r--', linewidth=1)

    # Trajetória
    xs, ys = zip(*path)
    ax.plot(xs, ys, color='lightgray', linewidth=1, linestyle='--', label="Traçado")            # Personalizar o percurso final na animação

    # Carro (retângulo que será atualizado)
    car_patch = ax.fill([], [], color='green', alpha=0.8)[0]                                      #Personalizar o carro na animação

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Animação realista do carro na pista")
    ax.legend()
    
    def get_car_polygon(x, y, heading, L=2.0, W=1.2):
        half_L = L / 2.0
        half_W = W / 2.0
        local_points = np.array([
            [ half_L,  half_W],
            [ half_L, -half_W],
            [-half_L, -half_W],
            [-half_L,  half_W]
        ])
        R = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading),  np.cos(heading)]
        ])
        global_points = (R @ local_points.T).T + np.array([x, y])
        return global_points

    def update(frame):
        i = frame
        
        # Garante que o índice não vá além do penúltimo ponto
        if i >= len(path) - 1:
            i = len(path) - 2
        
        x, y = path[i]
        x_next, y_next = path[i + 1]
        heading = np.arctan2(y_next - y, x_next - x)

        corners = get_car_polygon(x, y, heading, L, W)
        car_patch.set_xy(corners)
        return car_patch,

    anim = animation.FuncAnimation(fig, update, frames=len(path), interval=30, blit=False)

    if save_as:
        Writer = animation.writers['pillow']
        writer = Writer(fps=30)
        anim.save(save_as, writer=writer)
        print("GIF criado")
    else:
        plt.show()
    return anim


def run():
    start_time = time.time()
    F = 0.75
    CR = 0.85
    max_gen = 6000
    pop_size = 100
    track_start = 0
    track_end = 200
    num_plots = 20      #quantidade de plots desejados
    step= max_gen/num_plots
    gens_to_plot = generations_to_plot(num_plots, step)

    timestamp = datetime.now().strftime("%Y-%m-%d %H%M")
    results_dir = f"Resultados_Otimizacao_GLOBAL_Spline_{timestamp}_{max_gen}GEN_{pop_size}POP"
    os.makedirs(results_dir, exist_ok=True)

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
        folder=results_dir,
        use_parallel= True,
        n_jobs= 3
    )

    time_elapsed = time.time() - start_time
    hours, seconds_remain = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(seconds_remain, 60)
    time_ = hours,minutes,seconds
    print(f"Melhores parâmetros encontrados: {best_params}")
    print(f"Menor tempo total encontrada: {best_fitness}")
    print(f"Desvio padrão: {standard_deviation}")
    print(f"Total time elapsed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    save_results(best_params, best_fitness, standard_deviation, max_gen, pop_size,F,CR,time_,timestamp,results_dir)
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

    plt.savefig(f"{results_dir}/charts.png")
    # TRACK.plotar_tracado_na_pista(f"somulação {timstamp}/{fileName} tracado.png", best_params[0], track_size)

    
    
    path = generate_path(best_params[0], track_start=0, track_end=len(TRACK.CHECKPOINTS))
    animate_car_on_track_realistic(TRACK, path, best_params[1], L=2.0, W=1.2,save_as=f"{results_dir}/carro_na_pista.gif")

if __name__ ==   '__main__':
    run()

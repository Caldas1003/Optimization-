import numpy as np
from massa_mola_amortecedor import MassaMolaAmortecedor
from entrada_pista import entrada, trepidacao_pista

def funcao_custo(params, m=60, x0=[0, 0], track=None):
    k, c = params
    sistema = MassaMolaAmortecedor(m, k, c)

    positions = []
    velocities = []

    total_time = 0
    for i in range(len(track) - 1):
        t_span_segment = (total_time, total_time + 1)

        sol = sistema.simular(x0, t_span_segment, entrada, trepidacao_pista)

        positions.extend(sol.y[0, :])
        velocities.extend(sol.y[1, :])

        x0 = [sol.y[0, -1], sol.y[1, -1]]
        total_time += 1

    # Calcular o desvio padrão da posição
    position_std = np.std(positions)

    max_position = np.max(np.abs(positions))
    penalty_position = max_position ** 4

    damping_ratio = c / (2 * np.sqrt(k * m))

    # Penalidade se ξ estiver fora do intervalo [0, 1] (subamortecido)
    penalty_damping_ratio = 0
    if damping_ratio <= 0 or damping_ratio >= 1:
        penalty_damping_ratio = 1e6  

    # Penalidade adicional se ξ estiver muito baixo (muito próximo de 0)
    penalty_low_damping = 0
    if damping_ratio < 0.1:  
        penalty_low_damping = 1e3  # Penalidade moderada para valores muito baixos

    cost = position_std * penalty_position + penalty_damping_ratio + penalty_low_damping

    return cost


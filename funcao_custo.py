import numpy as np
from massa_mola_amortecedor import MassaMolaAmortecedor
from entrada_pista import entrada, trepidacao_pista


def funcao_custo(params, m=200, x0=[0, 10], track=None):
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
    penalty = max_position ** 4  # Penalidade para grandes oscilações

    # Ajuste o fator de penalização conforme necessário
    cost = position_std * penalty

    return cost

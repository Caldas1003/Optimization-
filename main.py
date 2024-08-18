import numpy as np
import matplotlib.pyplot as plt
from entrada_pista import entrada, trepidacao_pista
from massa_mola_amortecedor import MassaMolaAmortecedor
from differential_evolution import custom_differential_evolution
from gps_utils import gps_to_meters

def calcular_comprimento_pista(track):
    comprimento_total = 0
    for i in range(1, len(track)):
        dist_x = track[i][0] - track[i - 1][0]
        dist_y = track[i][1] - track[i - 1][1]
        comprimento_total += np.sqrt(dist_x**2 + dist_y**2)
    return comprimento_total

# Coordenadas GPS dos pontos-chave ao longo da pista (latitude, longitude)
gps_data = [
    (-22.738659, -47.533115),
    (-22.739670, -47.533586),
    (-22.740450, -47.533946),
    (-22.740695, -47.533828),
    (-22.740653, -47.533362),
    (-22.739632, -47.532706),
    (-22.739436, -47.532565),
    (-22.739348, -47.531401),
    (-22.738977, -47.530961),
    (-22.738267, -47.531071),
    (-22.737613, -47.531700),
    (-22.737403, -47.532262),
    (-22.737868, -47.532717),
    (-22.738659, -47.533115)
]

track = []
x = [0]
y = [0]

for i in range(1, len(gps_data)):
    dist_x = gps_to_meters(gps_data[i - 1][0], gps_data[i - 1][1], gps_data[i - 1][0], gps_data[i][1])
    dist_y = gps_to_meters(gps_data[i - 1][0], gps_data[i - 1][1], gps_data[i][0], gps_data[i - 1][1])

    if gps_data[i][1] < gps_data[i - 1][1]:
        dist_x = -dist_x
    if gps_data[i][0] < gps_data[i - 1][0]:
        dist_y = -dist_y

    x.append(x[-1] + dist_x)
    y.append(y[-1] + dist_y)
    track.append([x[-1], y[-1]])

comprimento_pista = calcular_comprimento_pista(track)

plt.figure(figsize=(10, 10))
plt.plot(x, y, marker='o', label='Traçado da Pista')
plt.title('Traçado Preciso da Pista')
plt.xlabel('Distância em X (metros)')
plt.ylabel('Distância em Y (metros)')
plt.grid(True)
plt.legend()
plt.show()

# Definir limites e estratégia para otimização
bounds = [(0.1, 300000), (0.1, 10000)]
strategy = "DE/rand/1"
best_params, best_cost = custom_differential_evolution(strategy, bounds, track, seed=1)

print("Parâmetros otimizados (k, c):", best_params)
print("Função de custo mínima:", best_cost)

sistema_otimizado = MassaMolaAmortecedor(m=250, k=best_params[0], c=best_params[1])
sol_otimizado = sistema_otimizado.simular([0, 0], (0, 100), entrada, trepidacao_pista)

plt.figure(figsize=(10, 5))
plt.plot(sol_otimizado.t, sol_otimizado.y[0], label='Posição (m)')
plt.plot(sol_otimizado.t, sol_otimizado.y[1], label='Velocidade (m/s)')
plt.title('Resposta Otimizada do Sistema Massa-Mola-Amortecedor')
plt.xlabel('Tempo (s)')
plt.ylabel('Resposta')
plt.legend()
plt.grid()
plt.show()



import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

# Função para converter diferenças de coordenadas GPS em metros
def gps_to_meters(lat1, lon1, lat2, lon2):
    R = 6371000  # Raio da Terra em metros
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distância em metros
    distance = R * c
    return distance

# Coordenadas GPS dos pontos-chave ao longo da pista (latitude, longitude)
gps_data = [
    (-22.738659, -47.533115),  # Ponto 1
    (-22.739670, -47.533586),  # Ponto 2
    (-22.740450, -47.533946),  # Ponto 3
    (-22.740695, -47.533828),  # Ponto 4
    (-22.740653, -47.533362),  # Ponto 5
    (-22.739632, -47.532706),  # Ponto 6
    (-22.739436, -47.532565),  # Ponto 7
    (-22.739348, -47.531401),  # Ponto 8
    (-22.738977, -47.530961),  # Ponto 9
    (-22.738267, -47.531071),  # Ponto 10
    (-22.737613, -47.531700),  # Ponto 11
    (-22.737403, -47.532262),  # Ponto 12
    (-22.737868, -47.532717),  # Ponto 13
    (-22.738659, -47.533115)   # Ligando Ponto 13 ao Ponto 1
]

# Altitudes associadas a cada ponto da pista (em metros)
altitudes = [
    0.0,  # Altitude para Ponto 1
    0.1,  # Altitude para Ponto 2
    0.1,  # Altitude para Ponto 3
    0.2,  # Altitude para Ponto 4
    0.1,  # Altitude para Ponto 5
    0.1,  # Altitude para Ponto 6
    0.0,  # Altitude para Ponto 7
    0.2,  # Altitude para Ponto 8
    0.1,  # Altitude para Ponto 9
    0.0,  # Altitude para Ponto 10
    0.1,  # Altitude para Ponto 11
    0.3,  # Altitude para Ponto 12
    0.0,  # Altitude para Ponto 13
    0.0   # Altitude para Ponto 1 (conectando ao ponto inicial)
]

# Convertendo coordenadas GPS para uma escala 2D em metros
x = [0]
y = [0]

for i in range(1, len(gps_data)):
    dist_x = gps_to_meters(gps_data[i - 1][0], gps_data[i - 1][1], gps_data[i - 1][0], gps_data[i][1])
    dist_y = gps_to_meters(gps_data[i - 1][0], gps_data[i - 1][1], gps_data[i][0], gps_data[i - 1][1])

    if gps_data[i][1] < gps_data[i - 1][1]:  # Ajuste para longitude (negativo se diminuir)
        dist_x = -dist_x
    if gps_data[i][0] < gps_data[i - 1][0]:  # Ajuste para latitude (negativo se diminuir)
        dist_y = -dist_y

    x.append(x[-1] + dist_x)
    y.append(y[-1] + dist_y)

# Certificando-se de que o primeiro e o último pontos são iguais
x.append(x[0])
y.append(y[0])
altitudes.append(altitudes[0])

# Criando um contorno paralelo para a pista
x_contorno = []
y_contorno = []

for i in range(len(x) - 1):
    dx = x[i + 1] - x[i]
    dy = y[i + 1] - y[i]
    length = np.sqrt(dx ** 2 + dy ** 2)

    # Vetores unitários perpendicular e deslocamento de 5 metros
    ux = -dy / length * 5
    uy = dx / length * 5

    x_contorno.append(x[i] + ux)
    y_contorno.append(y[i] + uy)

# Fechando o contorno conectando o último ponto ao primeiro
x_contorno.append(x_contorno[0])
y_contorno.append(y_contorno[0])

# Interpolando a pista e o contorno usando uma spline cúbica
t = np.linspace(0, 1, len(x))
spl_x = CubicSpline(t, x, bc_type='periodic')
spl_y = CubicSpline(t, y, bc_type='periodic')
spl_altitudes = CubicSpline(t, altitudes, bc_type='periodic')

t_new = np.linspace(0, 1, 200)
x_smooth = spl_x(t_new)
y_smooth = spl_y(t_new)
altitudes_smooth = spl_altitudes(t_new)

# Interpolando o contorno usando spline cúbica
spl_x_contorno = CubicSpline(t, x_contorno, bc_type='periodic')
spl_y_contorno = CubicSpline(t, y_contorno, bc_type='periodic')

x_contorno_smooth = spl_x_contorno(t_new)
y_contorno_smooth = spl_y_contorno(t_new)

# Plotando a pista e o contorno em 3D com superfície sólida
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# Preenchendo a área entre a pista original e o contorno
for i in range(len(x_smooth) - 1):
    x_fill = [x_smooth[i], x_smooth[i + 1], x_contorno_smooth[i + 1], x_contorno_smooth[i]]
    y_fill = [y_smooth[i], y_smooth[i + 1], y_contorno_smooth[i + 1], y_contorno_smooth[i]]
    z_fill = [altitudes_smooth[i], altitudes_smooth[i + 1], altitudes_smooth[i + 1], altitudes_smooth[i]]
    ax.plot_trisurf(x_fill, y_fill, z_fill, color='gray', alpha=1.0)

# Plotando as linhas da pista original e do contorno
ax.plot(x_smooth, y_smooth, altitudes_smooth, marker='o', color='black', label="Pista")
ax.plot(x_contorno_smooth, y_contorno_smooth, altitudes_smooth, marker='o', color='black')

# Definindo a escala dos eixos para manter as proporções corretas
max_range = max(max(x_smooth) - min(x_smooth), max(y_smooth) - min(y_smooth), max(altitudes_smooth) - min(altitudes_smooth))

# Calculando os centros dos eixos para centralizar a pista
mid_x = (max(x_smooth) + min(x_smooth)) / 2
mid_y = (max(y_smooth) + min(y_smooth)) / 2
mid_z = (max(altitudes_smooth) + min(altitudes_smooth)) / 2

ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

ax.set_title('Traçado da Pista com Superfície Sólida e Curvas Suaves')
ax.set_xlabel('Distância em X (metros)')
ax.set_ylabel('Distância em Y (metros)')
ax.set_zlabel('Altitude (metros)')
ax.legend()
ax.grid(True)

plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import CubicSpline
# from mpl_toolkits.mplot3d import Axes3D

# # Função para normalizar um vetor 
# def normalize(v):
#     norm = np.linalg.norm(v)
#     if norm == 0: 
#        return v
#     return v / norm

# # Função simples para converter latitude/longitude para coordenadas planas X, Y
# def latlon_to_xy(lat, lon, lat0, lon0):
#     R = 6371000  # raio da Terra em metros
#     x = R * np.radians(lon - lon0) * np.cos(np.radians(lat0))
#     y = R * np.radians(lat - lat0)
#     return x, y

# # Defina a largura da pista (6 metros para cada lado)
# pista_largura = 6

# # Coordenadas GPS coletadas (latitude, longitude)
# gps_coords = [
#     (-22.738762, -47.533146),
#     (-22.739971, -47.533735),
#     (-22.740344, -47.533928),
#     (-22.740598, -47.533945),
#     (-22.740725, -47.533782),
#     (-22.740737, -47.533451),
#     (-22.739432, -47.532570),
#     (-22.739353, -47.531387),
#     (-22.739159, -47.531069),
#     (-22.738715, -47.530897),
#     (-22.738259, -47.531082),
#     (-22.737450, -47.531959),
#     (-22.737394, -47.532273),
#     (-22.737490, -47.532471),
#     (-22.737608, -47.532600),
#     (-22.738504, -47.533038),
#     (-22.738762, -47.533146)  # Fechar o loop da pista
# ]

# # Definir um ponto de referência (ponto inicial para conversão)
# lat0, lon0 = gps_coords[0]

# # Converter as coordenadas GPS para X, Y
# coordenadas_centro = [latlon_to_xy(lat, lon, lat0, lon0) + (1,) for lat, lon in gps_coords]

# # Separar as coordenadas X, Y, Z
# X_centro = [p[0] for p in coordenadas_centro]
# Y_centro = [p[1] for p in coordenadas_centro]
# Z_centro = [p[2] for p in coordenadas_centro]

# # Criar uma interpolação cúbica para suavizar a pista sem a condição 'periodic'
# cs_x = CubicSpline(np.arange(len(X_centro)), X_centro, bc_type='natural')
# cs_y = CubicSpline(np.arange(len(Y_centro)), Y_centro, bc_type='natural')
# cs_z = CubicSpline(np.arange(len(Z_centro)), Z_centro, bc_type='natural')

# # Gerar mais pontos para suavizar a pista
# t = np.linspace(0, len(X_centro)-1, 500)
# X_suave = cs_x(t)
# Y_suave = cs_y(t)
# Z_suave = cs_z(t)

# # Função para calcular as bordas esquerda e direita da pista
# def calcular_bordas(X, Y, Z, largura_pista):
#     esquerda = []
#     direita = []
    
#     for i in range(len(X) - 1):
#         # Vetor direção no plano XY
#         vetor_direcao = np.array([X[i+1] - X[i], Y[i+1] - Y[i]])
        
#         # Vetor normal perpendicular
#         vetor_normal = np.array([-vetor_direcao[1], vetor_direcao[0]])
#         vetor_normal = normalize(vetor_normal)
        
#         # Calcular as bordas esquerda e direita
#         esquerda.append((X[i] + vetor_normal[0] * largura_pista, Y[i] + vetor_normal[1] * largura_pista, Z[i]))
#         direita.append((X[i] - vetor_normal[0] * largura_pista, Y[i] - vetor_normal[1] * largura_pista, Z[i]))
    
#     # Adicionar o último ponto para fechar as bordas
#     esquerda.append((X[-1], Y[-1], Z[-1]))
#     direita.append((X[-1], Y[-1], Z[-1]))
    
#     return esquerda, direita

# # Calcular as bordas da pista
# esquerda, direita = calcular_bordas(X_suave, Y_suave, Z_suave, pista_largura)

# # Separar as coordenadas X, Y, Z para as bordas esquerda e direita
# X_esq, Y_esq, Z_esq = zip(*esquerda)
# X_dir, Y_dir, Z_dir = zip(*direita)

# # Criar uma figura 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plotar a pista central suave
# ax.plot(X_suave, Y_suave, Z_suave, color='blue', label="Centro da Pista")

# # Plotar as bordas da pista
# ax.plot(X_esq, Y_esq, Z_esq, color='green', label="Borda Esquerda")
# ax.plot(X_dir, Y_dir, Z_dir, color='red', label="Borda Direita")

# # Conectar as bordas para fechar a pista
# for i in range(len(X_esq)):
#     ax.plot([X_esq[i], X_dir[i]], [Y_esq[i], Y_dir[i]], [Z_esq[i], Z_dir[i]], color='gray')

# # Configurar rótulos dos eixos
# ax.set_xlabel('X (metros)')
# ax.set_ylabel('Y (metros)')
# ax.set_zlabel('Z (altitude metros)')

# # Título
# ax.set_title('Simulação da Pista de Corrida Suavizada com Largura')

# # Exibir a grade e a legenda
# ax.grid(True)
# ax.legend()

# # Mostrar a simulação
# plt.show()


"PISTA COM ATRITO-MALHA" 

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D

# Função para normalizar um vetor 
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

# Função simples para converter latitude/longitude para coordenadas planas X, Y
def latlon_to_xy(lat, lon, lat0, lon0):
    R = 6371000  # raio da Terra em metros
    x = R * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = R * np.radians(lat - lat0)
    return x, y

# Defina a largura da pista (6 metros para cada lado)
pista_largura = 3
atrito = 1.45  # Coeficiente de atrito

# Coordenadas GPS coletadas (latitude, longitude)
gps_coords = [
    (-22.738762, -47.533146),
    (-22.739971, -47.533735),
    (-22.740344, -47.533928),
    (-22.740598, -47.533945),
    (-22.740725, -47.533782),
    (-22.740737, -47.533451),
    (-22.739432, -47.532570),
    (-22.739353, -47.531387),
    (-22.739159, -47.531069),
    (-22.738715, -47.530897),
    (-22.738259, -47.531082),
    (-22.737450, -47.531959),
    (-22.737394, -47.532273),
    (-22.737490, -47.532471),
    (-22.737608, -47.532600),
    (-22.738504, -47.533038),
    (-22.738762, -47.533146)  # Fechar o loop da pista
]

# Definir um ponto de referência (ponto inicial para conversão)
lat0, lon0 = gps_coords[0]

# Converter as coordenadas GPS para X, Y
coordenadas_centro = [latlon_to_xy(lat, lon, lat0, lon0) + (1,) for lat, lon in gps_coords]

# Separar as coordenadas X, Y, Z
X_centro = [p[0] for p in coordenadas_centro]
Y_centro = [p[1] for p in coordenadas_centro]
Z_centro = [p[2] for p in coordenadas_centro]

# Criar uma interpolação cúbica para suavizar a pista sem a condição 'periodic'
cs_x = CubicSpline(np.arange(len(X_centro)), X_centro, bc_type='natural')
cs_y = CubicSpline(np.arange(len(Y_centro)), Y_centro, bc_type='natural')
cs_z = CubicSpline(np.arange(len(Z_centro)), Z_centro, bc_type='natural')

# Gerar mais pontos para suavizar a pista
t = np.linspace(0, len(X_centro)-1, 200)
X_suave = cs_x(t)
Y_suave = cs_y(t)
Z_suave = cs_z(t)

# Função para calcular as bordas esquerda e direita da pista
def calcular_bordas(X, Y, Z, largura_pista):
    esquerda = []
    direita = []
    
    for i in range(len(X) - 1):
        # Vetor direção no plano XY
        vetor_direcao = np.array([X[i+1] - X[i], Y[i+1] - Y[i]])
        
        # Vetor normal perpendicular
        vetor_normal = np.array([-vetor_direcao[1], vetor_direcao[0]])
        vetor_normal = normalize(vetor_normal)
        
        # Calcular as bordas esquerda e direita
        esquerda.append((X[i] + vetor_normal[0] * largura_pista, Y[i] + vetor_normal[1] * largura_pista, Z[i]))
        direita.append((X[i] - vetor_normal[0] * largura_pista, Y[i] - vetor_normal[1] * largura_pista, Z[i]))
    
    # Adicionar o último ponto para fechar as bordas
    esquerda.append((X[-1], Y[-1], Z[-1]))
    direita.append((X[-1], Y[-1], Z[-1]))
    
    return esquerda, direita

# Calcular as bordas da pista
esquerda, direita = calcular_bordas(X_suave, Y_suave, Z_suave, pista_largura)

# Separar as coordenadas X, Y, Z para as bordas esquerda e direita
X_esq, Y_esq, Z_esq = zip(*esquerda)
X_dir, Y_dir, Z_dir = zip(*direita)

# Função para gerar a malha entre as bordas esquerda e direita
def gerar_malha(X_esq, Y_esq, Z_esq, X_dir, Y_dir, Z_dir, atrito):
    # Para criar uma malha, precisamos de uma matriz de coordenadas
    X_malha = np.array([X_esq, X_dir]), atrito
    Y_malha = np.array([Y_esq, Y_dir]), atrito
    Z_malha = np.array([Z_esq, Z_dir]), atrito
    return X_malha, Y_malha, Z_malha

# Gerar a malha
X_malha, Y_malha, Z_malha = gerar_malha(X_esq, Y_esq, Z_esq, X_dir, Y_dir, Z_dir, atrito) 

# Criar uma figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotar a pista central suave
ax.plot(X_suave, Y_suave, Z_suave, color='blue', label="Centro da Pista")

# Plotar as bordas da pista
ax.plot(X_esq, Y_esq, Z_esq, color='green', label="Borda Esquerda")
ax.plot(X_dir, Y_dir, Z_dir, color='red', label="Borda Direita")

# Conectar as bordas para fechar a pista
for i in range(len(X_esq)):
    ax.plot([X_esq[i], X_dir[i]], [Y_esq[i], Y_dir[i]], [Z_esq[i], Z_dir[i]], color='gray')

# Plotar a malha entre as bordas com o coeficiente de atrito
ax.plot_surface(X_malha[0], Y_malha[0], Z_malha[0], color='orange', alpha=0.6, label=f"Malha (Atrito {atrito})")

# Configurar rótulos dos eixos
ax.set_xlabel('X (metros)')
ax.set_ylabel('Y (metros)')
ax.set_zlabel('Z (altitude metros)')

# Título
ax.set_title('Simulação da Pista com Malha de Atrito')

# Exibir a grade e a legenda
ax.grid(True)
ax.legend()

# Mostrar a simulação
plt.show()

# Função para encontrar o ponto mais próximo na malha e verificar o atrito
def verificar_atrito(ponto, X_malha, Y_malha, Z_malha, atrito):
    """
    Verifica se o atrito está corretamente indexado para um ponto dado.
    
    Parâmetros:
    ponto (tuple): Um ponto (x, y, z) para verificar.
    X_malha, Y_malha, Z_malha (numpy arrays): Malha de pontos X, Y, Z.
    atrito (float): O valor do coeficiente de atrito aplicado.

    Retorna:
    dist_min (float): A menor distância encontrada para o ponto mais próximo.
    ponto_mais_proximo (tuple): O ponto da malha mais próximo do ponto dado.
    """
    x, y, z = ponto

    # Calcular a distância de cada ponto da malha até o ponto fornecido
    distancias = np.sqrt((X_malha - x) ** 2 + (Y_malha - y) ** 2 + (Z_malha - z) ** 2)
    
    # Encontrar a menor distância e o índice do ponto mais próximo
    indice_mais_proximo = np.unravel_index(np.argmin(distancias), distancias.shape)
    dist_min = distancias[indice_mais_proximo]
    
    # Obter as coordenadas do ponto mais próximo na malha
    ponto_mais_proximo = (X_malha[indice_mais_proximo], Y_malha[indice_mais_proximo], Z_malha[indice_mais_proximo])
    
    print(f"Coeficiente de atrito aplicado: {atrito}")
    print(f"Ponto mais próximo encontrado: {ponto_mais_proximo}")
    print(f"Distância mínima do ponto fornecido à malha: {dist_min:.4f} metros")
    
    return dist_min, ponto_mais_proximo

# Exemplo de ponto a ser verificado
ponto_teste = (228.1386, -2.8528, 1)

# Verificar o atrito no ponto fornecido
dist_min, ponto_mais_proximo = verificar_atrito(ponto_teste, X_malha[0], Y_malha[0], Z_malha[0], atrito)

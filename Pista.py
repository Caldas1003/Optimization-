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

# Defina a largura da pista (6 metros para cada lado)
pista_largura = 6

# Coordenadas centrais X, Y, Z coletadas
# Essas são as coordenadas centrais reais
# Era para ser um círculo mas o gepeto é burro, depois eu troco
coordenadas_centro = [
    (-200, 0, 10),
    (-150, 50, 12),
    (-100, 100, 8),
    (-50, 150, 5),
    (0, 200, 10),
    (50, 150, 12),
    (100, 100, 8),
    (150, 50, 5),
    (200, 0, 10),
    (150, -50, 12),
    (100, -100, 8),
    (50, -150, 5),
    (0, -200, 10),
    (-50, -150, 12),
    (-100, -100, 8),
    (-150, -50, 5)
]

# Separar as coordenadas X, Y, Z
X_centro = [p[0] for p in coordenadas_centro]
Y_centro = [p[1] for p in coordenadas_centro]
Z_centro = [p[2] for p in coordenadas_centro]

# Criar uma interpolação cúbica para suavizar a pista sem a condição 'periodic'
cs_x = CubicSpline(np.arange(len(X_centro)), X_centro, bc_type='natural')
cs_y = CubicSpline(np.arange(len(Y_centro)), Y_centro, bc_type='natural')
cs_z = CubicSpline(np.arange(len(Z_centro)), Z_centro, bc_type='natural')

# Gerar mais pontos para suavizar a pista
t = np.linspace(0, len(X_centro)-1, 500)
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

# Configurar rótulos dos eixos
ax.set_xlabel('X (metros)')
ax.set_ylabel('Y (metros)')
ax.set_zlabel('Z (altitude metros)')

# Título
ax.set_title('Simulação da Pista de Corrida Suavizada com Largura')

# Exibir a grade e a legenda
ax.grid(True)
ax.legend()

# Mostrar a simulação
plt.show()

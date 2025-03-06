"PISTA V4"

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

class PistaComAtrito:
    def __init__(self, gps_coords, elevacao, atrito, largura_pista=3):
        self.gps_coords = gps_coords
        self.elevacao = elevacao
        self.atrito = atrito
        self.largura_pista = largura_pista
        
        # Definir ponto de referência para conversão
        self.lat0, self.lon0 = gps_coords[0]
        
        # Converter as coordenadas GPS para X, Y e Z
        self.coordenadas_centro = [self.latlon_to_xy(lat, lon) + (1,) for lat, lon in gps_coords]
        self.X_centro = [p[0] for p in self.coordenadas_centro]
        self.Y_centro = [p[1] for p in self.coordenadas_centro]
        self.Z_centro = [e / 2 for e in elevacao]
        
        # Suavizar a pista usando interpolação cúbica
        self.suavizar_pista()

        # Calcular as bordas esquerda e direita da pista
        self.esquerda, self.direita = self.calcular_bordas()

        self.CHECKPOINTS = self.build_checkpoints()
    
    def latlon_to_xy(self, lat, lon):
        R = 6371000  # raio da Terra em metros
        x = R * np.radians(lon - self.lon0) * np.cos(np.radians(self.lat0))
        y = R * np.radians(lat - self.lat0)
        return x, y

    def suavizar_pista(self):
        t = np.linspace(0, len(self.X_centro) - 1, 200)
        cs_x = CubicSpline(np.arange(len(self.X_centro)), self.X_centro, bc_type='periodic')
        cs_y = CubicSpline(np.arange(len(self.Y_centro)), self.Y_centro, bc_type='periodic')
        cs_z = CubicSpline(np.arange(len(self.Z_centro)), self.Z_centro, bc_type='periodic')
        
        self.X_suave = cs_x(t)
        self.Y_suave = cs_y(t)
        self.Z_suave = cs_z(t)
    
    def normalize(self, v):
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm

    def calcular_bordas(self):
        esquerda = []
        direita = []
        
        for i in range(len(self.X_suave) - 1):
            vetor_direcao = np.array([self.X_suave[i + 1] - self.X_suave[i], self.Y_suave[i + 1] - self.Y_suave[i]])
            vetor_normal = self.normalize(np.array([-vetor_direcao[1], vetor_direcao[0]]))
            
            esquerda.append((self.X_suave[i] + vetor_normal[0] * self.largura_pista,
                             self.Y_suave[i] + vetor_normal[1] * self.largura_pista,
                             self.Z_suave[i]))
            direita.append((self.X_suave[i] - vetor_normal[0] * self.largura_pista,
                            self.Y_suave[i] - vetor_normal[1] * self.largura_pista,
                            self.Z_suave[i]))
        
        esquerda.append(esquerda[0])
        direita.append(direita[0])
        
        return esquerda, direita

    def gerar_malha(self):
        X_esq, Y_esq, Z_esq = zip(*self.esquerda)
        X_dir, Y_dir, Z_dir = zip(*self.direita)
        
        X_malha = np.array([X_esq, X_dir]), self.atrito
        Y_malha = np.array([Y_esq, Y_dir]), self.atrito
        Z_malha = np.array([Z_esq, Z_dir]), self.atrito
        return X_malha, Y_malha, Z_malha

    def verificar_atrito(self, ponto):
        X_malha, Y_malha, Z_malha = self.gerar_malha()
        x, y, z = ponto
        
        distancias = np.sqrt((X_malha[0] - x) ** 2 + (Y_malha[0] - y) ** 2 + (Z_malha[0] - z) ** 2)
        indice_mais_proximo = np.unravel_index(np.argmin(distancias), distancias.shape)
        dist_min = distancias[indice_mais_proximo]
        
        ponto_mais_proximo = (X_malha[0][indice_mais_proximo], Y_malha[0][indice_mais_proximo], Z_malha[0][indice_mais_proximo])
        
        print(f"Coeficiente de atrito aplicado: {self.atrito}")
        print(f"Ponto mais próximo encontrado: {ponto_mais_proximo}")
        print(f"Distância mínima do ponto fornecido à malha: {dist_min:.4f} metros")
        
        return dist_min, ponto_mais_proximo

    def plotar_pista(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        X_esq, Y_esq, Z_esq = zip(*self.esquerda)
        X_dir, Y_dir, Z_dir = zip(*self.direita)
        X_malha, Y_malha, Z_malha = self.gerar_malha()

        ax.plot(self.X_suave, self.Y_suave, self.Z_suave, color='blue', label="Centro da Pista")
        ax.plot(X_esq, Y_esq, Z_esq, color='green', label="Borda Esquerda")
        ax.plot(X_dir, Y_dir, Z_dir, color='red', label="Borda Direita")
        
        for i in range(len(X_esq)):
            ax.plot([X_esq[i], X_dir[i]], [Y_esq[i], Y_dir[i]], [Z_esq[i], Z_dir[i]], color='gray')
        
        ax.plot_surface(X_malha[0], Y_malha[0], Z_malha[0], color='black', alpha=1, label=f"Malha (Atrito {self.atrito})")
        
        ax.set_xlabel('X (metros)')
        ax.set_ylabel('Y (metros)')
        ax.set_zlabel('Z (altitude metros)')
        ax.set_title('Simulação da Pista com Malha de Atrito')
        ax.grid(True)
        ax.legend()
        
        plt.show()

    def build_checkpoints(self) -> list[list]:
        X_malha, Y_malha, Z_malha = self.gerar_malha()
        X_left = X_malha[0][0]
        X_right = X_malha[0][1]
        Y_left = Y_malha[0][0]
        Y_right = Y_malha[0][1]
        Z_left = Z_malha[0][0]
        Z_right = Z_malha[0][1]

        gates = list()
        for i, value in enumerate(X_left):
            left_point = (X_left[i], Y_left[i], Z_left[i])
            right_point = (X_right[i], Y_right[i], Z_right[i])

            num_points = 100
            t = np.linspace(0, 1, num_points)
            waypoints = np.outer(1 - t, left_point) + np.outer(
                t, right_point
            )  # 100 points between the left and right points

            gates.append(waypoints)

        return gates

    def plotar_tracado_na_pista(self, saveAs, waypoints, track_size=200):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        X_esq, Y_esq, Z_esq = zip(*self.esquerda)
        X_dir, Y_dir, Z_dir = zip(*self.direita)
        tracado_X, tracado_Y, tracado_Z = [], [], []

        for i, value in enumerate(waypoints):
            waypoint = self.CHECKPOINTS[i][value] # array de 3 posições (x, y, z)
            tracado_X.append(waypoint[0])
            tracado_Y.append(waypoint[1])
            tracado_Z.append(waypoint[2])

        ax.plot(tracado_X, tracado_Y, color='blue', label="Traçado", linewidth=0.5)
        ax.plot(X_esq[:track_size], Y_esq[:track_size], color='red', label="Borda Esquerda", linestyle='--', linewidth=0.5)
        ax.plot(X_dir[:track_size], Y_dir[:track_size], color='red', label="Borda Direita", linestyle='--', linewidth=0.5)
        
        # for i in range(len(X_esq)):
        #     ax.plot([X_esq[i], X_dir[i]], [Y_esq[i], Y_dir[i]], [Z_esq[i], Z_dir[i]], color='gray')
        
        ax.set_xlabel('X (metros)')
        ax.set_ylabel('Y (metros)')
        # ax.set_zlabel('Z (altitude metros)')
        ax.set_title('Exibição do traçado na pista')
        ax.grid(True)
        ax.legend()
        
        plt.savefig(saveAs)

    def plotar_parcial(self, parcial=50):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        X_esq, Y_esq, Z_esq = zip(*self.esquerda)
        X_dir, Y_dir, Z_dir = zip(*self.direita)
        tracado_X, tracado_Y, tracado_Z = [], [], []

        ax.plot(X_esq[:parcial], Y_esq[:parcial], color='red', label="Borda Esquerda")
        ax.plot(X_dir[:parcial], Y_dir[:parcial], color='red', label="Borda Direita")
        
        # for i in range(len(X_esq)):
        #     ax.plot([X_esq[i], X_dir[i]], [Y_esq[i], Y_dir[i]], [Z_esq[i], Z_dir[i]], color='gray')
        
        ax.set_xlabel('X (metros)')
        ax.set_ylabel('Y (metros)')
        # ax.set_zlabel('Z (altitude metros)')
        ax.set_title('parcial')
        ax.grid(True)
        ax.legend()
        
        plt.savefig("parcial")

# Definir coordenadas e elevacoes
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
    (-22.738762, -47.533146)
]
elevacao = [10.4, 11.8, 11.7, 10.8, 9.8, 8.3, 7.5, 1.9, 0.4, 0.1, 1.4, 5.0, 6.6, 7.5, 8.0, 10.0, 10.4]

# Criar a instância da pista com atrito
TRACK = PistaComAtrito(gps_coords, elevacao, atrito=1.45)

# Exemplo de ponto a ser verificado
ponto_teste = (228.1386, -2.8528, 1)
# pista.verificar_atrito(ponto_teste)

# # Plotar a pista
# TRACK.plotar_pista()

# TRACK.plotar_parcial(70)

import numpy as np
import matplotlib.pyplot as plt

class Pista:
    def __init__(self, gps_coords, largura_pista=6):
        self.gps_coords = gps_coords
        self.largura_pista = largura_pista

        # Definir ponto de referência para conversão
        self.lat0, self.lon0 = gps_coords[0]

        # Converter as coordenadas GPS para X e Y
        self.coordenadas_centro = [self.latlon_to_xy(lat, lon) for lat, lon in gps_coords]
        self.X_centro = [p[0] for p in self.coordenadas_centro]
        self.Y_centro = [p[1] for p in self.coordenadas_centro]

        # Calcular as bordas esquerda e direita da pista
        self.esquerda, self.direita = self.calcular_bordas()

    def latlon_to_xy(self, lat, lon):
        R = 6371000  # raio da Terra em metros
        x = R * np.radians(lon - self.lon0) * np.cos(np.radians(self.lat0))
        y = R * np.radians(lat - self.lat0)
        return x, y

    def normalize(self, v):
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm

    def calcular_bordas(self):
        esquerda = []
        direita = []

        n = len(self.X_centro)
        for i in range(n):
            # Obter os índices do ponto anterior e do próximo (cíclico)
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n

            # Vetor direção entre o ponto anterior e o próximo
            vetor_direcao = np.array([self.X_centro[next_idx] - self.X_centro[prev_idx],
                                      self.Y_centro[next_idx] - self.Y_centro[prev_idx]])
            vetor_normal = self.normalize(np.array([-vetor_direcao[1], vetor_direcao[0]]))

            # Calcular as bordas com base na largura da pista
            esquerda.append((self.X_centro[i] + vetor_normal[0] * self.largura_pista / 2,
                             self.Y_centro[i] + vetor_normal[1] * self.largura_pista / 2))
            direita.append((self.X_centro[i] - vetor_normal[0] * self.largura_pista / 2,
                            self.Y_centro[i] - vetor_normal[1] * self.largura_pista / 2))

        return esquerda, direita

    def plotar_pista(self):
        fig, ax = plt.subplots()

        X_esq, Y_esq = zip(*self.esquerda)
        X_dir, Y_dir = zip(*self.direita)

        # Traçar a pista
        ax.plot(self.X_centro, self.Y_centro, color='blue', label="Centro da Pista")
        ax.plot(X_esq, Y_esq, color='green', label="Borda Esquerda")
        ax.plot(X_dir, Y_dir, color='red', label="Borda Direita")

        # Traçar as retas conectando as bordas
        for i in range(len(X_esq)):
            ax.plot([X_esq[i], X_dir[i]], [Y_esq[i], Y_dir[i]], color='gray', linewidth=0.5)

        # Adicionar as linhas de largada e chegada
        ax.plot([X_esq[0], X_dir[0]], [Y_esq[0], Y_dir[0]], color='black', linestyle='--', label='Linha de Largada')
        ax.plot([X_esq[-1], X_dir[-1]], [Y_esq[-1], Y_dir[-1]], color='purple', linestyle='--', label='Linha de Chegada')

        ax.set_xlabel('X (metros)')
        ax.set_ylabel('Y (metros)')
        ax.set_title('Simulação da Pista')
        ax.grid(True)
        ax.legend()

        plt.axis('equal')
        plt.show()

# Coordenadas GPS de exemplo
gps_coords = [
    (-22.738674, -47.533101),
    (-22.738810, -47.533171),
    (-22.740274, -47.533893),
    (-22.740422, -47.533944),
    (-22.740595, -47.533935),
    (-22.740704, -47.533817),
    (-22.740757, -47.533727),
    (-22.740757, -47.533568),
    (-22.740707, -47.533415),
    (-22.739457, -47.532591),
    (-22.739448, -47.532462),
    (-22.739348, -47.531396),
    (-22.739221, -47.531150),
    (-22.739011, -47.530965),
    (-22.738653, -47.530897),
    (-22.738491, -47.530954),
    (-22.738281, -47.531067),
    (-22.737617, -47.531700),
    (-22.737422, -47.532007),
    (-22.737395, -47.532266),
    (-22.737480, -47.532472),
    (-22.737598, -47.532587),
    (-22.738110, -47.532835),
    (-22.738501, -47.533029),
    (-22.738674, -47.533101)
]

# Criar e plotar a pista
pista = Pista(gps_coords, largura_pista=10)
pista.plotar_pista()

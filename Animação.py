import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
from Pista import Pista, gps_coords

class SimulacaoCarro:
    def __init__(self, pista, sentido='largada_para_chegada'):
        self.pista = pista
        self.checkpoints = []
        self.portoes = []
        self.tracado = []
        self.angulos = []
        self.sentido = sentido

    def gerar_checkpoints(self, num_checkpoints=49, num_portoes=10):
        t = np.linspace(0, len(self.pista.X_centro) - 1, num_checkpoints, dtype=int)
        for i in t:
            x_esq, y_esq = self.pista.esquerda[i]
            x_dir, y_dir = self.pista.direita[i]
            checkpoint = {'esquerda': (x_esq, y_esq), 'direita': (x_dir, y_dir)}
            self.checkpoints.append(checkpoint)

            # Criar portões igualmente espaçados com margem de segurança
            margem = 0.15
            portoes_checkpoint = []
            for j in range(num_portoes):
                alfa = margem + j * (1 - 2 * margem) / (num_portoes - 1)
                x = x_esq + alfa * (x_dir - x_esq)
                y = y_esq + alfa * (y_dir - y_esq)
                portoes_checkpoint.append((x, y))
            self.portoes.append(portoes_checkpoint)

    def selecionar_portao_aleatorio(self):
        """Seleciona um portão aleatório dependendo do sentido."""
        if self.sentido == 'largada_para_chegada':
            return random.choice(self.portoes[0])  # Linha de largada
        else:
            return random.choice(self.portoes[-1])  # Linha de chegada

    def simular_tracado(self, angulo_maximo=10):
        portao_atual = self.selecionar_portao_aleatorio()
        self.tracado.append(portao_atual)

        if self.sentido == 'largada_para_chegada':
            for i in range(len(self.portoes) - 1):
                portoes_atual = self.portoes[i]
                portoes_prox = self.portoes[i + 1]

                distancias = [np.linalg.norm(np.array(portao_atual) - np.array(portao)) for portao in portoes_prox]
                angulos_validos = []

                for j, portao in enumerate(portoes_prox):
                    if len(self.tracado) > 1:
                        p_anterior = np.array(self.tracado[-2])
                        p_atual = np.array(self.tracado[-1])
                        direcao_anterior = p_atual - p_anterior
                        direcao_atual = np.array(portao) - np.array(portao_atual)

                        if np.linalg.norm(direcao_anterior) == 0 or np.linalg.norm(direcao_atual) == 0:
                            continue

                        cos_theta = np.dot(direcao_anterior, direcao_atual) / (np.linalg.norm(direcao_anterior) * np.linalg.norm(direcao_atual))
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        angulo = np.degrees(np.arccos(cos_theta))

                        if angulo <= angulo_maximo:
                            angulos_validos.append((j, distancias[j]))

                if angulos_validos:
                    indice_portao_prox = min(angulos_validos, key=lambda x: x[1])[0]
                else:
                    indice_portao_prox = np.argmin(distancias)

                portao_atual = portoes_prox[indice_portao_prox]
                self.tracado.append(portao_atual)

        elif self.sentido == 'chegada_para_largada':
            for i in range(len(self.portoes) - 1, 0, -1):
                portoes_atual = self.portoes[i]
                portoes_prox = self.portoes[i - 1]

                distancias = [np.linalg.norm(np.array(portao_atual) - np.array(portao)) for portao in portoes_prox]
                angulos_validos = []

                for j, portao in enumerate(portoes_prox):
                    if len(self.tracado) > 1:
                        p_anterior = np.array(self.tracado[-2])
                        p_atual = np.array(self.tracado[-1])
                        direcao_anterior = p_atual - p_anterior
                        direcao_atual = np.array(portao) - np.array(portao_atual)

                        if np.linalg.norm(direcao_anterior) == 0 or np.linalg.norm(direcao_atual) == 0:
                            continue

                        cos_theta = np.dot(direcao_anterior, direcao_atual) / (np.linalg.norm(direcao_anterior) * np.linalg.norm(direcao_atual))
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        angulo = np.degrees(np.arccos(cos_theta))

                        if angulo <= angulo_maximo:
                            angulos_validos.append((j, distancias[j]))

                if angulos_validos:
                    indice_portao_prox = min(angulos_validos, key=lambda x: x[1])[0]
                else:
                    indice_portao_prox = np.argmin(distancias)

                portao_atual = portoes_prox[indice_portao_prox]
                self.tracado.append(portao_atual)

    def analise_angulos(self):
        angulos = []
        for i in range(1, len(self.tracado) - 1):
            p_anterior = np.array(self.tracado[i - 1])
            p_atual = np.array(self.tracado[i])
            p_proximo = np.array(self.tracado[i + 1])

            direcao_anterior = p_atual - p_anterior
            direcao_atual = p_proximo - p_atual

            if np.linalg.norm(direcao_anterior) == 0 or np.linalg.norm(direcao_atual) == 0:
                continue

            cos_theta = np.dot(direcao_anterior, direcao_atual) / (
                    np.linalg.norm(direcao_anterior) * np.linalg.norm(direcao_atual))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angulo = np.degrees(np.arccos(cos_theta))
            angulos.append(angulo)

        if not angulos:
            return 0, 0, 0  # Retornar valores padrão

        max_ang = max(angulos)
        min_ang = min(angulos)
        mean_ang = sum(angulos) / len(angulos)

        return max_ang, min_ang, mean_ang

    def plotar_pista_e_tracado(self):
        fig, ax = plt.subplots()

        X_esq, Y_esq = zip(*self.pista.esquerda)
        X_dir, Y_dir = zip(*self.pista.direita)

        # Plotar a pista
        ax.plot(self.pista.X_centro, self.pista.Y_centro, color='blue', label="Centro da Pista")
        ax.plot(X_esq, Y_esq, color='green', label="Borda Esquerda")
        ax.plot(X_dir, Y_dir, color='red', label="Borda Direita")

        # Plotar o traçado escolhido
        x_tracado, y_tracado = zip(*self.tracado)
        ax.plot(x_tracado, y_tracado, color='orange', label="Traçado Escolhido", linewidth=2)

        ax.set_xlabel('X (metros)')
        ax.set_ylabel('Y (metros)')
        ax.set_title('Pista e Traçado Simulado')
        ax.legend()
        plt.axis('equal')
        plt.show()

    def animacao(self):
        fig, ax = plt.subplots()

        X_esq, Y_esq = zip(*self.pista.esquerda)
        X_dir, Y_dir = zip(*self.pista.direita)

        ax.plot(self.pista.X_centro, self.pista.Y_centro, color='blue', label="Centro da Pista")
        ax.plot(X_esq, Y_esq, color='green', label="Borda Esquerda")
        ax.plot(X_dir, Y_dir, color='red', label="Borda Direita")

        ponto_particula, = ax.plot([], [], 'ro', label="Partícula")
        linha_tracado, = ax.plot([], [], color='orange', label="Traçado")

        def init():
            ponto_particula.set_data([], [])
            linha_tracado.set_data([], [])
            return ponto_particula, linha_tracado

        def update(frame):
            if frame < len(self.tracado):
                ponto_particula.set_data([self.tracado[frame][0]],
                                         [self.tracado[frame][1]])  # Passar listas para set_data
                linha_tracado.set_data([p[0] for p in self.tracado[:frame + 1]],
                                       [p[1] for p in self.tracado[:frame + 1]])
            return ponto_particula, linha_tracado

        anim = FuncAnimation(fig, update, frames=range(len(self.tracado)), init_func=init, blit=True, interval=1000)

        plt.xlabel('X (metros)')
        plt.ylabel('Y (metros)')
        plt.title('Movimento da Partícula e Traçado')
        plt.legend()
        plt.axis('equal')
        plt.show()

# Instanciar a pista e a simulação
pista = Pista(gps_coords, largura_pista=6)
simulacao = SimulacaoCarro(pista, sentido='largada_para_chegada')

# Executar a simulação
simulacao.gerar_checkpoints()
simulacao.simular_tracado(angulo_maximo=10)

# Gerar animação do movimento
simulacao.animacao()

# Instanciar a pista e a simulação
pista = Pista(gps_coords, largura_pista=6)
simulacao = SimulacaoCarro(pista, sentido='chegada_para_largada')

# Executar a simulação
simulacao.gerar_checkpoints()
simulacao.simular_tracado(angulo_maximo=10)

# Gerar animação do movimento
simulacao.animacao()

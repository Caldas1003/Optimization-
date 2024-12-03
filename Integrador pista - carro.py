import numpy as np
import matplotlib.pyplot as plt
from PistaComAtrito import PistaComAtrito,gps_coords, elevacao

class SimulacaoCarro:
    def __init__(self, pista):
        self.pista = pista
        self.checkpoints = []
        self.portoes = []
        self.tracado = []
        self.angulos = []

    def gerar_checkpoints(self, num_checkpoints=10000, num_portoes=100):
        t = np.linspace(0, len(self.pista.X_suave) - 1, num_checkpoints)
        for i in range(len(t) - 1):
            x_esq, y_esq, z_esq = self.pista.esquerda[int(t[i])]
            x_dir, y_dir, z_dir = self.pista.direita[int(t[i])]
            checkpoint = {'esquerda': (x_esq, y_esq, z_esq), 'direita': (x_dir, y_dir, z_dir)}
            self.checkpoints.append(checkpoint)

            # Criar portões igualmente espaçados
            portoes_checkpoint = []
            for j in range(num_portoes):
                alfa = j / (num_portoes - 1)
                x = x_esq + alfa * (x_dir - x_esq)
                y = y_esq + alfa * (y_dir - y_esq)
                z = z_esq + alfa * (z_dir - z_esq)
                portoes_checkpoint.append((x, y, z))
            self.portoes.append(portoes_checkpoint)

    def simular_tracado(self, angulo_maximo=6):
        portao_atual = self.portoes[0][0]  # Começa no primeiro portão
        self.tracado.append(portao_atual)

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
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plotar a pista
        X_esq, Y_esq, Z_esq = zip(*self.pista.esquerda)
        X_dir, Y_dir, Z_dir = zip(*self.pista.direita)
        ax.plot(self.pista.X_suave, self.pista.Y_suave, self.pista.Z_suave, color='blue', label="Centro da Pista")
        ax.plot(X_esq, Y_esq, Z_esq, color='green', label="Borda Esquerda")
        ax.plot(X_dir, Y_dir, Z_dir, color='red', label="Borda Direita")

        # Plotar o traçado escolhido
        x_tracado, y_tracado, z_tracado = zip(*self.tracado)
        ax.plot(x_tracado, y_tracado, z_tracado, color='orange', label="Traçado Escolhido", linewidth=2)

        ax.set_xlabel('X (metros)')
        ax.set_ylabel('Y (metros)')
        ax.set_zlabel('Z (altitude metros)')
        ax.set_title('Pista e Traçado Simulado')
        ax.legend()
        plt.show()

pista = PistaComAtrito(gps_coords, elevacao, atrito=1.45)
simulacao = SimulacaoCarro(pista)

simulacao.gerar_checkpoints()
simulacao.simular_tracado(angulo_maximo=20)

try:
    max_ang, min_ang, mean_ang = simulacao.analise_angulos()
    print(f"Máximo ângulo: {max_ang:.2f}°, Mínimo ângulo: {min_ang:.2f}°, Média dos ângulos: {mean_ang:.2f}°")
except ValueError as e:
    print(f"Erro na análise de ângulos: {e}")

simulacao.plotar_pista_e_tracado()


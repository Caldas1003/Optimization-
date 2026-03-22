import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import os


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
        self.Z_centro = [e / e for e in elevacao]
        
        # Suavizar a pista usando interpolação cúbica
        self.suavizar_pista()

        # Calcular as bordas esquerda e direita da pista
        self.esquerda, self.direita = self.calcular_bordas()

        self.construir_arvore_busca()

        self.CHECKPOINTS = self.build_checkpoints()
    
    def latlon_to_xy(self, lat, lon):
        R = 6371000
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

    def construir_arvore_busca(self):
        points_track = np.column_stack((self.X_suave, self.Y_suave))
        self.tree = cKDTree(points_track)

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
        
        #ax.plot_surface(X_malha[0], Y_malha[0], Z_malha[0], color='black', alpha=1, label=f"Malha (Atrito {self.atrito})")
        
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

            num_points = 10
            t = np.linspace(0, 1, num_points)
            waypoints = np.outer(1 - t, left_point) + np.outer(
                t, right_point
            )  #  points between the left and right points

            gates.append(waypoints)

        return gates

    def plotar_tracado_na_pista(self, saveAs, path, track_size=100, checkpoints=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        X_esq, Y_esq, Z_esq = zip(*self.esquerda)
        X_dir, Y_dir, Z_dir = zip(*self.direita)
        tracado_X, tracado_Y = [], []

        for point in path:
            tracado_X.append(point[0])
            tracado_Y.append(point[1])

        ax.plot(tracado_X, tracado_Y, color='blue', label="Traçado", linewidth=0.5)
        ax.plot(X_esq[:track_size], Y_esq[:track_size], color='red', label="Borda Esquerda", linestyle='--', linewidth=0.5)
        ax.plot(X_dir[:track_size], Y_dir[:track_size], color='red', label="Borda Direita", linestyle='--', linewidth=0.5)
        
        if checkpoints:
            for i in range(track_size):
                ax.plot([X_esq[i], X_dir[i]], [Y_esq[i], Y_dir[i]], [Z_esq[i], Z_dir[i]], color='gray')
        
            for i in range(len(path)):
                if i % 10 == 0:
                    ax.plot(tracado_X[i], tracado_Y[i], 'o', color='red')
        ax.set_xlabel('X (metros)')
        ax.set_ylabel('Y (metros)')
        # ax.set_zlabel('Z (altitude metros)')
        # ax.set_xlim(-90, 10)
        # ax.set_ylim(-250, 10)
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

    def plotar_tracado_na_pista_com_velocidade(self, saveAs: str, path: list, speed_profile: list, track_start=0, track_end=100, checkpoints=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        X_esq, Y_esq, Z_esq = zip(*self.esquerda)
        X_dir, Y_dir, Z_dir = zip(*self.direita)

        tracado_2D = []

        for point in path:
            tracado_2D.append([point[0], point[1]])
                
        tracado_2D = np.array(tracado_2D)
        speed_profile = np.array(speed_profile)

        if checkpoints:
            for i in range(track_start, track_end):
                ax.plot([X_esq[i], X_dir[i]], [Y_esq[i], Y_dir[i]], color='gray')
        
            for i in range(len(path)):
                if i % 10 == 0:
                    ax.plot(tracado_2D[i][0], tracado_2D[i][1], 'o', color='red', markersize=1)

        segments = np.concatenate([tracado_2D[:-1, None], tracado_2D[1:, None]], axis=1)
        lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(speed_profile.min(), speed_profile.max()))
        lc.set_array(speed_profile[:-1])
        lc.set_linewidth(0.8)

        ax.plot(X_esq[track_start:track_end], Y_esq[track_start:track_end], color='red', label="Borda Esquerda", linestyle='--', linewidth=0.5)
        ax.plot(X_dir[track_start:track_end], Y_dir[track_start:track_end], color='red', label="Borda Direita", linestyle='--', linewidth=0.5)
        
        # for i in range(len(X_esq)):
        #     ax.plot([X_esq[i], X_dir[i]], [Y_esq[i], Y_dir[i]], [Z_esq[i], Z_dir[i]], color='gray')
        
        ax.set_xlabel('X (metros)')
        ax.set_ylabel('Y (metros)')
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label='Gradient values')
        # ax.set_zlabel('Z (altitude metros)')
        ax.set_title('Exibição do traçado na pista')
        ax.grid(True)
        # ax.set_xlim(-90, 10)
        # ax.set_ylim(-250, 10)
        ax.legend()
        
        plt.savefig(saveAs)
        fig.clear()

    def closest_point_on_center(self, x, y):
        _, idx = self.tree.query([x, y])
        return self.X_suave[idx], self.Y_suave[idx], self.Z_suave[idx]       

def save_track_to_npy(track_object: PistaComAtrito, track_name: str, output_dir: str = 'tracks'):
    """
    Combina as coordenadas suavizadas da Linha Central e das Bordas
    no formato (N, 6) e salva como um arquivo .npy.

    Formato (N, 6): [x_c, y_c, x_i, y_i, x_o, y_o]
    """
    
    # Extrai X e Y das bordas. A borda `direita` será usada como `interna` e `esquerda` como `externa`
    # Você pode inverter se a orientação da sua pista for diferente.
    X_esq, Y_esq, _ = zip(*track_object.esquerda)
    X_dir, Y_dir, _ = zip(*track_object.direita)

    # N é o número de waypoints suavizados. O último ponto em X_suave é o primeiro, então N = len(X_suave)
    N_suave = len(track_object.X_suave) 

    # 1. Linha Central (Center Line): Usar os arrays suaves (N_suave pontos)
    center_arr = np.column_stack((track_object.X_suave, track_object.Y_suave))

    # 2. Borda Interna (Inner Border): Usar os arrays da direita/esquerda (N_suave pontos)
    # As coordenadas de X_dir/Y_dir e X_esq/Y_esq têm N_suave + 1 pontos (o último é duplicado).
    # Usamos apenas os primeiros N_suave pontos únicos.
    inner_arr = np.column_stack((X_dir[:N_suave], Y_dir[:N_suave])) 
    outer_arr = np.column_stack((X_esq[:N_suave], Y_esq[:N_suave])) 

    # 3. Empilhar as colunas para formar o array (N_suave, 6)
    # [x_c, y_c, x_i, y_i, x_o, y_o]
    waypoints_6d = np.hstack((center_arr, inner_arr, outer_arr))

    # 4. Fechar o loop adicionando o primeiro ponto no final (total N_suave + 1 pontos)
    # Isso é o que o script de raceline espera
    primeiro_ponto = waypoints_6d[0].reshape(1, 6)
    waypoints_final = np.vstack((waypoints_6d, primeiro_ponto))
    
    # 5. Salvar o arquivo
    track_file_name = f"{track_name}.npy"
    output_path = os.path.join(output_dir, track_file_name)

    os.makedirs(output_dir, exist_ok=True)
    np.save(output_path, waypoints_final)

    print("\n--- Geração do Arquivo .npy ---")
    print(f"Pista: {track_name}")
    print(f"Número de Waypoints Salvos (Loop Fechado): {len(waypoints_final)}")
    print(f"Array salvo com sucesso em: {output_path}")
    print("---------------------------------")
    
    return output_path

# Funções para gerar diferentes tipos de pista
def gerar_pista_original():
    
    gps_coords = [
    (-22.738923621, -47.533307707),        #01             
    (-22.740234945, -47.533971047),        #02
    #(-22.740471163, -47.534002315),        #03 
    #(-22.740572611, -47.533970103),        #04 
    (-22.740630346, -47.533888070),        #05
    #(-22.740714497, -47.533762456),        #06
    (-22.740755167, -47.533660327),        #07
    #(-22.740737214, -47.533516870),        #08
    (-22.740576845, -47.533288432),        #09
    (-22.739544000, -47.532582070),        #10
    (-22.739400000, -47.531950000),        #11
    (-22.739353770, -47.531354061),        #12
    (-22.739242067, -47.531138614),        #13
    (-22.739006929, -47.530985476),        #14
    (-22.738711312, -47.530908197),        #15
    (-22.738356737, -47.531030504),        #17
    (-22.737526001, -47.531812107),        #18
    (-22.737379845, -47.532440590),        #20
    (-22.737553717, -47.532629243),        #21
    (-22.738513990, -47.533111324),        #22
    (-22.738923621, -47.533307707)         #23
        ]
    
    elevacao = [541.6, 544.5,  541.2,  537.4, 535.3, 534.2, 523.3, 521.3, 520.3, 521.2, 522.1, 522.5, 527.6, 531.1, 535.3, 538.4, 540.3, 541.6]

    return PistaComAtrito(gps_coords, elevacao, atrito=1.45)

def gerar_pista_interna():

    # Pista Interna
    gps_coords = [
    (-22.738893449, -47.533220318),
    (-22.740311956, -47.533920324),
    (-22.740679217, -47.533861951),
    (-22.740714268, -47.533404065),
    (-22.739420979, -47.532572491),
    (-22.738664639, -47.532148215),
    (-22.738374618, -47.532302949),
    (-22.738545709, -47.532565721),
    (-22.738770301, -47.532546194),
    (-22.738976343, -47.532629411),
    (-22.73993502, -47.533230206),
    (-22.740063205, -47.53340129),
    (-22.739812718, -47.533542839),
    (-22.738044537, -47.532601892),
    (-22.737849406, -47.532216989),
    (-22.738548273, -47.53139417),
    (-22.738903133, -47.531402565),
    (-22.738924653, -47.531916278),
    (-22.739390073, -47.531950908),
    (-22.739343093, -47.531377857),
    (-22.739102689, -47.53100597),
    (-22.738701893, -47.53089727),
    (-22.738292309, -47.531040372),
    (-22.73753052, -47.53179826),
    (-22.737363183, -47.532241553),
    (-22.737637175, -47.532621053),
    # Ponto inicial repetido
    (-22.738893449, -47.533220318)
    ]
        
    elevacao = [
    540.8, 543.9, 540.4, 535.0, 534.7, 530.9, 532.2, 535.7, 535.5, 536.6, 
    540.2, 542.0, 543.0, 537.1, 532.2, 524.6, 526.1, 531.8, 529.6, 523.6, 
    519.9, 521.1, 522.2, 527.5, 532.3, 538.3, 
    # Ponto inicial repetido
    540.8
    ]

    return PistaComAtrito(gps_coords, elevacao, atrito=1.45)

def gerar_pista_reta(comprimento=500, pontos=50):
    x_vals = np.linspace(0, comprimento, pontos)
    gps_coords = [(0.0 + y * 1e-5, -47.533000) for y in x_vals]
    return PistaComAtrito(gps_coords, atrito=1.2)

def gerar_pista_circular(raio=100, pontos=36):
    centro_lat, centro_lon = -22.738762, -47.533146
    angulos = np.linspace(0, 2*np.pi, pontos, endpoint=False)
    gps_coords = []
    for theta in angulos:
        dlat = (raio * np.sin(theta)) / 6371000 * (180 / np.pi)
        dlon = (raio * np.cos(theta)) / (6371000 * np.cos(np.radians(centro_lat))) * (180 / np.pi)
        gps_coords.append((centro_lat + dlat, centro_lon + dlon))
    return PistaComAtrito(gps_coords, atrito=1.3)


# Escolher tipo de pista
tipo_pista = "original"  # "original", "reta", "circular","interna"

if tipo_pista == "original":
    TRACK = gerar_pista_original()
elif tipo_pista == "reta":
    TRACK = gerar_pista_reta()
elif tipo_pista == "interna":
    TRACK = gerar_pista_interna()
elif tipo_pista == "circular":
    TRACK = gerar_pista_circular()
else:
    raise ValueError("Tipo de pista inválido")

track_name = f"{tipo_pista}_custom"

#save_track_to_npy(TRACK, track_name)

def xy_to_latlon(x, y, lat0, lon0):
    """
    Converte coordenadas x, y de volta para latitude e longitude.
    
    Parâmetros:
    x (float): Coordenada x em metros
    y (float): Coordenada y em metros
    lat0 (float): Latitude de referência (em graus) usada na conversão original
    lon0 (float): Longitude de referência (em graus) usada na conversão original
    
    Retorna:
    tuple: (latitude, longitude) em graus
    """
    R = 6371000  # Raio da Terra em metros
    
    # Converter a latitude de referência para radianos
    lat0_rad = np.radians(lat0)
    
    # Calcular a diferença de longitude (em radianos)
    delta_lon_rad = x / (R * np.cos(lat0_rad))
    
    # Calcular a diferença de latitude (em radianos)
    delta_lat_rad = y / R
    
    # Converter as diferenças para graus
    delta_lon = np.degrees(delta_lon_rad)
    delta_lat = np.degrees(delta_lat_rad)
    
    # Calcular as coordenadas finais
    lat = lat0 + delta_lat
    lon = lon0 + delta_lon
    
    return lat, lon

# Plotar a pista selecionada
#TRACK.plotar_pista()

"""
data =TRACK.coordenadas_centro
#print(TRACK.coordenadas_centro)
# Extrair coordenadas X e Y
x = [point[0] for point in data]
y = [point[1] for point in data]

# Criar o gráfico
plt.figure(figsize=(12, 8))
plt.plot(x, y, 'b-', marker='o', markersize=8)  # Linha azul com marcadores

# Adicionar números indicando a ordem dos pontos
for i, (xi, yi, _) in enumerate(data):
    plt.annotate(str(i+1),  # Texto (número do ponto)
                 (xi, yi),  # Coordenadas do ponto
                 textcoords="offset points",  # Sistema de coordenadas do offset
                 xytext=(0, 10),  # Deslocamento do texto (10 pontos acima)
                 ha='center',  # Alinhamento horizontal centralizado
                 fontsize=12,
                 fontweight='bold',
                 color='red')

plt.title("Gráfico dos Pontos com Numeração", fontsize=14)
plt.xlabel("Coordenada X", fontsize=12)
plt.ylabel("Coordenada Y", fontsize=12)
plt.grid(True, alpha=0.3)
plt.axis('equal')  # Mantém proporções iguais nos eixos

# Adicionar uma tabela com informações dos pontos
column_labels = ['Ponto', 'X', 'Y']
table_data = [[i+1, round(x[i], 2), round(y[i], 2)] for i in range(len(data))]

# Posicionar a tabela à direita do gráfico
plt.table(cellText=table_data,
          colLabels=column_labels,
          loc='right',
          bbox=[1.1, 0.1, 0.5, 0.8])

plt.tight_layout()
plt.show()
"""

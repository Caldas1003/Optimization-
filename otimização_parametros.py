import numpy as np
from pista import PistaComAtrito

class CarroGenerico:
    def __init__(self, forca_pedal, coef_atrito, psi, torque_motor, massa_carro, raio_rodas, rigidez_mola, amortecimento, tensao_bateria, corrente_bateria, capacidade_bateria):
        self.forca_pedal = forca_pedal
        self.coef_atrito = coef_atrito
        self.psi = psi
        
        self.torque_motor = torque_motor
        self.massa_carro = massa_carro
        self.raio_rodas = raio_rodas 
        
        self.rigidez_mola = rigidez_mola
        self.amortecimento = amortecimento
        
        self.tensao_bateria = tensao_bateria
        self.corrente_bateria = corrente_bateria
        self.capacidade_bateria = capacidade_bateria
    
    def calcular_tempo_frenagem(self, velocidade_inicial):
        desaceleracao = self.forca_pedal * self.coef_atrito * self.psi
        tempo_frenagem = velocidade_inicial / desaceleracao
        return np.maximum(tempo_frenagem, 0.1)
    
    def calcular_aceleracao(self):
        forca_rodas = self.torque_motor / self.raio_rodas
        aceleracao = forca_rodas / self.massa_carro
        return aceleracao

    def calcular_energia_bateria(self):
        energia_bateria = self.capacidade_bateria * 3600 * 1000
        return energia_bateria

    def calcular_energia_consumida(self, tempo_segmento):
        potencia = self.tensao_bateria * self.corrente_bateria
        energia_consumida = potencia * tempo_segmento
        return energia_consumida
    

    def calcular_tempo_segmento(self, distancia_segmento, velocidade_inicial, velocidade_final):
        velocidade_final_m_s = velocidade_final / 3.6
        velocidade_inicial_m_s = velocidade_inicial / 3.6

        aceleracao = self.calcular_aceleracao()
        tempo_frenagem = self.calcular_tempo_frenagem(velocidade_inicial_m_s)

        efeito_suspensao = 1 + (1 / (self.rigidez_mola + self.amortecimento))

        if aceleracao > 0: 
            tempo_aceleracao = (velocidade_final_m_s - velocidade_inicial_m_s) / aceleracao
            distancia_percorrida_acelerando = 0.5 * aceleracao * (tempo_aceleracao ** 2)
            distancia_percorrida_acelerando = distancia_percorrida_acelerando[:199]

            if isinstance(distancia_segmento, np.ndarray):
                # Ajuste o comprimento de distancia_segmento para corresponder ao comprimento de distancia_percorrida_acelerando
                if distancia_percorrida_acelerando.shape[0] < distancia_segmento.shape[0]:
                    distancia_segmento = distancia_segmento[:distancia_percorrida_acelerando.shape[0]]
                distancia_restante = distancia_segmento - distancia_percorrida_acelerando
                # Garantir que todas as variáveis sejam arrays de pelo menos uma dimensão
                tempo_aceleracao = np.atleast_1d(tempo_aceleracao)
                distancia_restante = np.atleast_1d(distancia_restante)
                velocidade_final_m_s = np.atleast_1d(velocidade_final_m_s)
                tempo_frenagem = np.atleast_1d(tempo_frenagem)

                # Verificar os tamanhos
                tamanho_minimo = min(tempo_aceleracao.shape[0], distancia_restante.shape[0], velocidade_final_m_s.shape[0], tempo_frenagem.shape[0])

                # Calcular o tempo
                tempo_total = (tempo_aceleracao[:tamanho_minimo] + 
                                (distancia_restante[:tamanho_minimo] / velocidade_final_m_s[:tamanho_minimo]) + 
                                tempo_frenagem[:tamanho_minimo]) * efeito_suspensao
            else:
                if distancia_percorrida_acelerando < distancia_segmento:
                    distancia_restante = distancia_segmento - distancia_percorrida_acelerando
                    tempo_total = (tempo_aceleracao + (distancia_restante / velocidade_final_m_s) + tempo_frenagem) * efeito_suspensao
                else:
                    tempo_total = (2 * distancia_segmento / aceleracao) ** 0.5 * efeito_suspensao
        else:
            tempo_total = (distancia_segmento / velocidade_final_m_s) * efeito_suspensao

        return tempo_total

# Função objetivo: minimizar o tempo de volta e considerar o impacto da bateria
def funcao_objetivo(params, pista, velocidade_inicial=72):
    forca_pedal, coef_atrito, psi, torque_motor, massa_carro, raio_rodas, rigidez_mola, amortecimento, tensao_bateria, corrente_bateria, capacidade_bateria = params

    carro = CarroGenerico(forca_pedal, coef_atrito, psi, torque_motor, massa_carro, raio_rodas, rigidez_mola, amortecimento, tensao_bateria, corrente_bateria, capacidade_bateria)
    coef_atrito = pista.atrito
    velocidade_inicial_ms = velocidade_inicial / 3.6
    tempo_total_volta = 0
    energia_total_consumida = 0
    torque_min = bounds[0][0]  # O valor mínimo do torque a partir dos limites
    psi_min_ideal = 0.5  # Valor mínimo aceitável para Psi (distribuição de frenagem)
    
    energia_disponivel = carro.calcular_energia_bateria()
    
    X_malha, Y_malha, Z_malha = pista.gerar_malha()
    distancia_total = np.sqrt(np.diff(X_malha[0])**2 + np.diff(Y_malha[0])**2 + np.diff(Z_malha[0])**2)

    for i in range(len(distancia_total)):
        distancia_segmento = distancia_total[i]
        velocidade_segmento = Z_malha[0][i] * 14
        tempo_segmento = carro.calcular_tempo_segmento(distancia_segmento, velocidade_inicial_ms, velocidade_segmento / 3.6)
        energia_consumida_segmento = carro.calcular_energia_consumida(tempo_segmento)
        
        # Verificar a forma de energia_consumida_segmento e garantir que a soma seja compatível
        energia_consumida_segmento = np.atleast_1d(energia_consumida_segmento)
        # Realizar a soma corretamente
        energia_total_consumida += np.sum(energia_consumida_segmento)
        # Verificar a forma de energia_consumida_segmento e garantir que a soma seja compatível
        tempo_segmento = np.atleast_1d(tempo_segmento)
        # Realizar a soma corretamente
        tempo_total_volta += np.sum(tempo_segmento)
        
        velocidade_inicial_ms = velocidade_segmento / 3.6

    if energia_total_consumida > energia_disponivel:
        penalidade = (energia_total_consumida / energia_disponivel)
        tempo_total_volta *= penalidade
    
    if psi < psi_min_ideal:
        # Penalidade exponencial para valores de Psi abaixo do ideal
        penalidade = 1 + (psi_min_ideal - psi) ** 3  # Penalidade cúbica
        tempo_total_volta *= penalidade
    
    if torque_motor < torque_min:
        penalidade = 1 + ((torque_min - torque_motor) / torque_min) ** 2  # Penalidade quadrática
        tempo_total_volta *= penalidade
    
    return tempo_total_volta

# Evolução Diferencial Auto-Adaptativa (EDA)
def evolucao_diferencial_adaptativa(func, pista, bounds, pop_size=110, F=0.8, CR=0.9, max_generations=1000):
    num_params = bounds.shape[0]
    pop = np.random.rand(pop_size, num_params)
    pop_denorm = np.array([bounds[:, 0] + p * (bounds[:, 1] - bounds[:, 0]) for p in pop])
    
    fitness = np.array([func(ind, pista) for ind in pop_denorm])
    
    for gen in range(max_generations):
        for i in range(pop_size):
            indices = list(range(pop_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            
            diff = pop_denorm[b] - pop_denorm[c]
            trial = pop_denorm[i] + F * diff
            trial = np.clip(trial, bounds[:, 0], bounds[:, 1])
            
            j_rand = np.random.randint(num_params)
            trial_vec = np.array([trial[j] if np.random.rand() < CR or j == j_rand else pop_denorm[i][j] for j in range(num_params)])
            
            trial_fitness = func(trial_vec, pista)
            if np.all(trial_fitness < fitness[i]):
                pop_denorm[i] = trial_vec
                fitness[i] = trial_fitness
        
        pop = np.array([(ind - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) for ind in pop_denorm])

        F = np.clip(F * np.random.rand(), 0.4, 1.2)
        CR = np.clip(CR * np.random.rand(), 0.1, 1.0)

        if np.std(fitness) < 1e-10:
            print(f"Convergência alcançada na geração {gen}.")
            break
        gen += 1
        fitness = np.where(np.isfinite(fitness), fitness, np.inf)
        best_idx = np.argmin(fitness)
        best_params = pop_denorm[best_idx]
        best_fitness = fitness[best_idx]
        best_params = np.clip(best_params, bounds[:, 0], bounds[:, 1])
    
    return best_params, best_fitness

# Configuração da pista e otimização
gps_coords = [  (-22.738762, -47.533146),
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
    (-22.738762, -47.533146)]  # Substitua com coordenadas reais
elevacao = [10.4, 11.8, 11.7, 10.8, 9.8, 8.3, 7.5, 1.9, 0.4, 0.1, 1.4, 5.0, 6.6, 7.5, 8.0, 10.0, 10.4]  # Substitua com dados reais de elevação
atrito = 1.45  # Exemplo de atrito

pista = PistaComAtrito(gps_coords, elevacao, atrito)
bounds = np.array([
    [445, 823],  # Força no pedal de freio (N)
    [0.1, 0.45],  # Coeficiente de atrito das pastilhas
    [0.3, 0.7],  # Distribuição da força de frenagem
    [50, 300],   # Torque do motor (Nm)
    [150, 200],  # Massa do carro (kg)
    [0.2, 0.35],  # Raio das rodas (m)
    [10000, 1000000],  # Rigidez da mola (N/m)
    [1000, 40000],  # Amortecimento da suspensão (Ns/m)
    [83, 666],  # Tensão da bateria (V)
    [150, 600],  # Corrente da bateria (A)
    [5, 10]  # Capacidade do pack de bateria (kWh)
])

# Executar a EDA para otimizar o tempo de volta
melhor_solucao, melhor_tempo_volta = evolucao_diferencial_adaptativa(funcao_objetivo, pista, bounds)

def tempo():
    if melhor_tempo_volta > 60:
        minuto = melhor_tempo_volta // 60
        segundo = melhor_tempo_volta % 60
        if minuto >= 60:
            hora = minuto // 60
            minuto = minuto % 60
        else:
            hora = 0
    else:
        minuto = 0
        segundo = melhor_tempo_volta
        hora = 0
    print(f'Melhor Tempo de Volta: {hora:.0f} h {minuto:.0f} m {segundo:.2f} s')

# Exibir a melhor solução e o melhor tempo de volta
print(f"Melhor Solução: Força no pedal = {melhor_solucao[0]:.2f} N, Coeficiente de atrito = {melhor_solucao[1]:.2f}, Psi = {melhor_solucao[2]:.2f}, Torque do motor = {melhor_solucao[3]:.2f} Nm, Massa = {melhor_solucao[4]:.2f} kg, Raio das rodas = {melhor_solucao[5]:.2f} m, Rigidez da mola: {melhor_solucao[6]:.2f} N/m, Amortecimento da suspensão: {melhor_solucao[7]:.2f} Ns/m, Tensão da bateria: {melhor_solucao[8]:.2f} V, Corrente da bateria: {melhor_solucao[9]:.2f} A, Capacidade do pack de bateria: {melhor_solucao[10]:.2f} kWh")
tempo()
print(f'Melhor Tempo de Volta em Segundos: {melhor_tempo_volta:.2f} s')

import numpy as np

# Classe genérica do sistema do carro
class CarroGenerico:
    def __init__(self, forca_pedal, coef_atrito, psi, torque_motor, massa_carro, raio_rodas, rigidez_mola, amortecimento, coef_aerodinamico, coef_atrito_pneu, tensao_bateria, corrente_bateria, capacidade_bateria):
        # Sistema de freios
        self.forca_pedal = forca_pedal  # Força no pedal de freio (N)
        self.coef_atrito = coef_atrito  # Coeficiente de atrito das pastilhas
        self.psi = psi  # Distribuição da força de frenagem (dianteira/traseira)
        
        # Sistema de transmissão
        self.torque_motor = torque_motor  # Torque do motor (Nm)
        self.massa_carro = massa_carro  # Massa do carro (kg)
        self.raio_rodas = raio_rodas  # Raio das rodas (m)
        
        # Suspensão
        self.rigidez_mola = rigidez_mola  # Rigidez da mola (N/m)
        self.amortecimento = amortecimento  # Amortecimento da suspensão (Ns/m) <-- SUBCRITICO
        
        # Aerodinâmica e atrito
        self.coef_aerodinamico = coef_aerodinamico  # Coeficiente de arrasto
        self.coef_atrito_pneu = coef_atrito_pneu  # Coeficiente de atrito do pneu com o solo
        
        # Sistema de bateria e eletrônica
        self.tensao_bateria = tensao_bateria  # Tensão (V)
        self.corrente_bateria = corrente_bateria  # Corrente elétrica (A)
        self.capacidade_bateria = capacidade_bateria  # Capacidade do pack de bateria (kWh)

        # Chassi
        # tensão maxima e minima
        # rigidez (esperar) transformar em ax + bu 
    
    def calcular_tempo_frenagem(self, velocidade_inicial):
        # Desaceleração gerada pelo freio
        desaceleracao = self.forca_pedal * self.coef_atrito * self.psi
        tempo_frenagem = velocidade_inicial / desaceleracao  # Tempo para parar
        return max(tempo_frenagem, 0.1)  # Garantir tempo positivo
    
    def calcular_aceleracao(self):
        # Força nas rodas = torque / raio
        forca_rodas = self.torque_motor / self.raio_rodas
        # Aceleração = Força / Massa
        aceleracao = forca_rodas / self.massa_carro
        return aceleracao

    def calcular_resistencia_ar(self, velocidade):
        # Resistência aerodinâmica = 0.5 * coef_aerodinamico * velocidade^2
        resistencia_ar = 0.5 * self.coef_aerodinamico * (velocidade ** 2)
        return resistencia_ar
    
    def calcular_energia_bateria(self):
        # Energia disponível na bateria (em Joules)
        energia_bateria = self.capacidade_bateria * 3600 * 1000  # kWh para Joules
        return energia_bateria

    def calcular_energia_consumida(self, tempo_segmento):
        # Energia consumida = Potência (tensão * corrente) * tempo
        potencia = self.tensao_bateria * self.corrente_bateria  # em Watts
        energia_consumida = potencia * tempo_segmento  # em Joules
        return energia_consumida
    
    def calcular_tempo_segmento(self, segmento, velocidade_inicial):
        # Parâmetros do segmento
        distancia_segmento = segmento['distancia']
        velocidade_final_m_s = segmento['velocidade'] / 3.6  # km/h para m/s
        
        # Calcular aceleração e frenagem
        aceleracao = self.calcular_aceleracao()
        tempo_frenagem = self.calcular_tempo_frenagem(velocidade_inicial)
        
        # Calcular o efeito da suspensão no tempo
        efeito_suspensao = 1 + (1 / (self.rigidez_mola + self.amortecimento))
        
        # Calcular o efeito da resistência do ar no tempo
        resistencia_ar = self.calcular_resistencia_ar(velocidade_final_m_s)
        efeito_aero = 1 + (resistencia_ar / 1000)  # Simulando o efeito da resistência
        
        # Calcular o tempo para percorrer o segmento
        if aceleracao > 0:
            tempo_aceleracao = (velocidade_final_m_s - velocidade_inicial) / aceleracao
            distancia_percorrida_acelerando = 0.5 * aceleracao * (tempo_aceleracao ** 2)
            if distancia_percorrida_acelerando < distancia_segmento:
                distancia_restante = distancia_segmento - distancia_percorrida_acelerando
                tempo_total = (tempo_frenagem + tempo_aceleracao + distancia_restante / velocidade_final_m_s) * efeito_suspensao * efeito_aero
            else:
                tempo_total = (2 * distancia_segmento / aceleracao) ** 0.5 * efeito_suspensao * efeito_aero
        else:
            tempo_total = (distancia_segmento / velocidade_final_m_s) * efeito_suspensao * efeito_aero

        return tempo_total


# Função objetivo: minimizar o tempo de volta e considerar o impacto da bateria
def funcao_objetivo(params, pista, velocidade_inicial=72):
    # Desempacotar os parâmetros
    forca_pedal, coef_atrito, psi, torque_motor, massa_carro, raio_rodas, rigidez_mola, amortecimento, coef_aerodinamico, coef_atrito_pneu, tensao_bateria, corrente_bateria, capacidade_bateria = params

    # Criar uma instância genérica do carro com os parâmetros
    carro = CarroGenerico(forca_pedal, coef_atrito, psi, torque_motor, massa_carro, raio_rodas, rigidez_mola, amortecimento, coef_aerodinamico, coef_atrito_pneu, tensao_bateria, corrente_bateria, capacidade_bateria)

    velocidade_inicial_ms = velocidade_inicial / 3.6  # km/h para m/s
    tempo_total_volta = 0
    energia_total_consumida = 0  # Energia consumida por volta
    energia_disponivel = carro.calcular_energia_bateria()

    # Calcular o tempo total e a energia consumida para todos os segmentos da pista
    for segmento in pista:
        tempo_segmento = carro.calcular_tempo_segmento(segmento, velocidade_inicial_ms)
        energia_consumida_segmento = carro.calcular_energia_consumida(tempo_segmento)
        
        energia_total_consumida += energia_consumida_segmento
        tempo_total_volta += tempo_segmento
        
        # Atualizar a velocidade inicial para o próximo segmento
        velocidade_inicial_ms = segmento['velocidade'] / 3.6

    # Se a energia consumida for maior que a disponível, o carro não completa a volta
    if energia_total_consumida > energia_disponivel:
        penalidade = (energia_total_consumida / energia_disponivel)  # Penalidade proporcional ao excesso de consumo
        tempo_total_volta *= penalidade

    return tempo_total_volta  # Queremos minimizar o tempo total de volta


# Evolução Diferencial Auto-Adaptativa (EDA)
def evolucao_diferencial_adaptativa(func, bounds, pista, pop_size=130, F=0.8, CR=0.9, max_generations=1000):
    dim = len(bounds)
    pop = np.random.rand(pop_size, dim)
    pop_denorm = bounds[:, 0] + pop * (bounds[:, 1] - bounds[:, 0])
    fitness = np.array([func(ind, pista) for ind in pop_denorm])

    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    log_tempo_voltas = []  # Armazenar o melhor tempo de cada geração
    log_tempo_voltas.append(fitness[best_idx])

    gen = 0
    while gen < max_generations:
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            r1, r2, r3 = pop[np.random.choice(idxs, 3, replace=False)]
            mutante = r1 + F * (r2 - r3)
            mutante = np.clip(mutante, 0, 1)
            cruzado = np.where(np.random.rand(dim) < CR, mutante, pop[i])
            cruzado_denorm = bounds[:, 0] + cruzado * (bounds[:, 1] - bounds[:, 0])
            f = func(cruzado_denorm, pista)
            if f < fitness[i]:
                fitness[i] = f
                pop[i] = cruzado
                if f < fitness[best_idx]:
                    best_idx = i
                    best = cruzado_denorm

        # Auto-adaptação de F e CR
        F = np.clip(F * np.random.rand(), 0.4, 1.2)
        CR = np.clip(CR * np.random.rand(), 0.1, 1.0)

        if np.std(fitness) < 1e-10:
            print(f"Convergência alcançada na geração {gen}.")
            break
        gen += 1

    return best, fitness[best_idx]

# Definir os limites dos parâmetros de controle (tudo em um único array)
bounds = np.array([
    [100, 823],  # Força no pedal de freio (N)
    [0.1, 0.45],  # Coeficiente de atrito das pastilhas
    [0.3, 0.7],  # Distribuição da força de frenagem
    [50, 300],   # Torque do motor (Nm)
    [150, 300],  # Massa do carro (kg)
    [0.2, 0.35],  # Raio das rodas (m)
    [10000, 1000000],  # Rigidez da mola (N/m)
    [1000, 40000],  # Amortecimento da suspensão (Ns/m)
    [0.25, 0.45],  # Coeficiente aerodinâmico
    [1.0, 1.45],  # Coeficiente de atrito do pneu com o solo
    [83, 666],  # Tensão da bateria (V)
    [150, 600],  # Corrente da bateria (A)
    [5, 10]  # Capacidade do pack de bateria (kWh)
])


# Exemplo de pista (definida em segmentos de distância e velocidade média)
pista = [
    {'distancia': 1000, 'velocidade': 90},  # Segmento 1
    {'distancia': 2000, 'velocidade': 108},  # Segmento 2
    {'distancia': 3000, 'velocidade': 126}  # Segmento 3
]

# Executar a EDA para otimizar o tempo de volta
melhor_solucao, melhor_tempo_volta = evolucao_diferencial_adaptativa(funcao_objetivo, bounds, pista)

def tempo():
    if melhor_tempo_volta > 60:
        minuto = melhor_tempo_volta // 60
        segundo = melhor_tempo_volta % 60
        if minuto > 60:
            hora = minuto / 60
        else:
            hora = 0
    else: 
        minuto = 0
    print(f'Melhor Tempo de Volta: {hora:.1f} h {minuto:.1f} m {segundo:.1f} s')

# Exibir a melhor solução e o melhor tempo de volta
print(f"Melhor Solução: Força no pedal = {melhor_solucao[0]:.2f} N, Coeficiente de atrito = {melhor_solucao[1]:.2f}, Psi = {melhor_solucao[2]:.2f}, Torque do motor = {melhor_solucao[3]:.2f} Nm, Massa = {melhor_solucao[4]:.2f} kg, Raio das rodas = {melhor_solucao[5]:.2f} m, Rigidez da mola: {melhor_solucao[6]:.2f} N/m, Amortecimento da suspensão: {melhor_solucao[7]:.2f} Ns/m, Coeficiente aerodinâmico: {melhor_solucao[8]:.2f}, Coeficiente de atrito do pneu com o solo: {melhor_solucao[9]:.2f}, Tensão da bateria: {melhor_solucao[10]:.2f} V, Corrente da bateria: {melhor_solucao[11]:.2f} A, Capacidade do pack de bateria: {melhor_solucao[12]:.2f} kWh")
tempo()
print(f'Melhor Tempo de Volta em Segundos: {melhor_tempo_volta:.2f} s')

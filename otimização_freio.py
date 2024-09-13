import numpy as np

# Classe genérica do sistema de freios
class BrakeSystem:
    def __init__(self, forca_pedal, coef_atrito, psi):
        self.forca_pedal = forca_pedal  # Força no pedal de freio (N)
        self.coef_atrito = coef_atrito  # Coeficiente de atrito das pastilhas
        self.psi = psi  # Distribuição da força de frenagem (dianteira/traseira)

    def calcular_tempo_frenagem(self, velocidade_inicial):
        # Cálculo genérico do tempo de frenagem com base nos parâmetros
        desaceleracao = self.forca_pedal * self.coef_atrito  # Desaceleração gerada pelo freio
        tempo_frenagem = velocidade_inicial / desaceleracao  # Tempo para parar completamente
        return max(tempo_frenagem, 0.1)  # Garantir que não haja tempo de frenagem negativo

# Função objetivo: minimizar o tempo de volta
def funcao_objetivo(params, pista, velocidade_inicial=72):
    # Desempacotar os parâmetros do vetor params (vindo da evolução diferencial)
    forca_pedal, coef_atrito, psi = params

    # Criar uma instância do sistema de freios com esses parâmetros
    sistema_freio = BrakeSystem(forca_pedal, coef_atrito, psi)
    velocidade_inicial_ms = velocidade_inicial / 3.6

    # Calcular o tempo total de volta
    tempo_total = 0
    for segmento in pista:
        # Para cada segmento, calcula-se o tempo de frenagem e aceleração
        tempo_frenagem = sistema_freio.calcular_tempo_frenagem(velocidade_inicial_ms)
        tempo_aceleracao = segmento['distancia'] / (segmento['velocidade']/3.6)  # Tempo para completar o segmento
        tempo_total += tempo_frenagem + tempo_aceleracao

    return tempo_total  # Queremos minimizar o tempo total de volta

# Evolução Diferencial Auto-adaptativa (EDA)
def evolucao_diferencial_adaptativa(func, bounds, pista, pop_size=20, F=0.8, CR=0.9, max_generations=1000):
    dim = len(bounds)
    pop = np.random.rand(pop_size, dim)
    pop_denorm = bounds[:, 0] + pop * (bounds[:, 1] - bounds[:, 0])
    fitness = np.array([func(ind, pista) for ind in pop_denorm])  

    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

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

# Definir os limites dos parâmetros de controle
bounds = np.array([
    [100, 823],  # Limites para a força no pedal de freio (N)
    [0.1, 0.45],  # Limites para o coeficiente de atrito das pastilhas
    [0.3, 0.7]  # Limites para a distribuição da força de frenagem (dianteira/traseira)
])

# Exemplo de pista (definida em segmentos de distância e velocidade média)
pista = [
    {'distancia': 1000, 'velocidade': 90},  # Segmento 1
    {'distancia': 2000, 'velocidade': 108},  # Segmento 2
    {'distancia': 3000, 'velocidade': 126}  # Segmento 3
]

# Parâmetros iniciais (para exemplo)
params_iniciais = [500, 0.4, 0.5]

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


print(f"Melhor Solução: Força no pedal = {melhor_solucao[0]:.2f} N, Coeficiente de atrito = {melhor_solucao[1]:.2f}, Psi = {melhor_solucao[2]:.2f}")
tempo()
print(f'Melhor Tempo de Volta em Segundos: {melhor_tempo_volta:.2f} s')

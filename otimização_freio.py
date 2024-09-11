# exemplo 1
import numpy as np

class BrakeSystem:
    # Inicializa a classe com os parâmetros necessários
    def __init__(self, params):
        self.params = params
        self.RedP = params['RedP']  # Redução do pedal
        self.a = params['a']  # Desaceleração [g]
        self.psi = params['psi']  # Distribuição de carga estática por eixo [adm]
        self.μl = params['μl']  # Coeficiente de atrito do contato pastilha/disco
        self.pi = params['pi']  # Valor de pi
        self.HCG = params['HCG']  # Altura do centro de gravidade [m]
        self.μ = params['μ']  # Coeficiente de atrito do pneu/solo
        self.FzF = params['FzF']  # Força de reação estática na dianteira [N]
        self.FzR = params['FzR']  # Força de reação estática na traseira [N]
        self.Rdp = params['Rdp']  # Raio do pneu [m]
        self.Dcm = params['Dcm']  # Diâmetro do cilindro mestre em metros
        self.Dwc = params['Dwc']  # Diâmetro do cilindro da roda em metros
        self.Npast = params['Npast']  # Número de pastilhas por disco
        self.atrito_coeficiente = params['atrito_coeficiente']  # Coeficiente de atrito da pastilha
        self.red = params['red']  # Raio efetivo do disco [m]
        self.Mt = params['Mt']  # Massa total do veículo [kg]
        self.L = params['L']  # Distância entre eixos [m]
        self.c_rr = params['c_rr']  # Coeficiente de resistência ao rolamento
        self.m_tire = params['m_tire']  # Massa do pneu
        self.m_wheel = params['m_wheel']  # Massa da roda

    # Calcula diversos parâmetros relacionados à frenagem
    def calculate_params(self, pedal_force=300):
        BF = 2 * self.μl  # Fator de freio
        χ = self.HCG / self.L  # Razão entre HCG e L
        W = self.Mt * 9.81  # Peso do veículo
        FzF_dyn = (1 - self.psi + self.a * χ) * W  # Força de reação dinâmica dianteira
        FzR_dyn = (self.psi - self.a * χ) * W  # Força de reação dinâmica traseira
        τF = FzF_dyn * self.μ * self.Rdp  # Torque na dianteira
        τR = FzR_dyn * self.μ * self.Rdp  # Torque na traseira
        FnF = τF / self.Npast * self.RedP * self.red  # Força normal das pastilhas dianteira
        FnR = τR / self.Npast * self.RedP * self.red  # Força normal das pastilhas traseira
        Awc = (self.pi * (self.Dwc ** 2)) / 4  # Área do cilindro de roda
        Acm = (self.pi * (self.Dcm ** 2)) / 4  # Área do cilindro mestre
        PF = FnF / Awc  # Pressão necessária na dianteira
        PR = FnR / Awc  # Pressão necessária na traseira
        FaCM = PF * Acm  # Força necessária para acionar o cilindro mestre
        return BF, FzF_dyn, FzR_dyn, τF, τR, FnF, FnR, Awc, Acm, PF, PR, FaCM

    # Aplica o freio e calcula a força de frenagem
    def apply_brake(self, pedal_force=300, coef_atrito=None):
        if coef_atrito:
            self.atrito_coeficiente = coef_atrito  # Atualiza o coeficiente de atrito

        resultados = self.calculate_params(pedal_force)
        BF, FzF_dyn, FzR_dyn, τF, τR, FnF, FnR, Awc, Acm, PF, PR, FaCM = resultados

        pressao_cilindro_mestre = pedal_force / Acm
        pressao_fluido = pressao_cilindro_mestre
        transmissao_pressao = pressao_fluido * Awc
        forca_aperto_pinca = transmissao_pressao
        forca_atrito_pastilha = forca_aperto_pinca * self.atrito_coeficiente
        torque_disco_freio = forca_atrito_pastilha * self.red
        resistencia_rolamento = self.c_rr * self.Mt * 9.81
        torque_resistencia_rolamento = resistencia_rolamento * self.Rdp
        torque_ajustado = torque_disco_freio - torque_resistencia_rolamento
        return torque_ajustado

    # Calcula a velocidade angular das rodas durante a frenagem
    def calculate_angular_velocity(self, torque_ajustado):
        time_intervals = np.linspace(0, tempo, 100)
        inertia_wheel = 0.5 * (self.m_tire + self.m_wheel) * (self.Rdp ** 2)
        angular_desaceleration = (torque_ajustado / 4) / inertia_wheel
        initial_angular_velocity = initial_speed / self.Rdp
        time_step = time_intervals[1] - time_intervals[0]
        angular_velocity = initial_angular_velocity
        angular_velocities = []

        for i in time_intervals:
            angular_velocity -= angular_desaceleration * time_step
            angular_velocities.append(angular_velocity)

        return angular_desaceleration, angular_velocity, angular_velocities, inertia_wheel

# Função para calcular o tempo de frenagem com base na desaceleração angular das rodas
def calcular_tempo_frenagem_angular(brake_system, torque_ajustado):
    angular_desaceleration, angular_velocity, angular_velocities, _ = brake_system.calculate_angular_velocity(torque_ajustado)
    tempo_frenagem = 0

    for v_angular in angular_velocities:
        if v_angular <= 0:
            break
        tempo_frenagem += tempo / len(angular_velocities)

    return tempo_frenagem

# Função objetivo para otimização: minimizar o tempo de frenagem
def func_objetivo_frenagem_angular(pedal_force, coef_atrito, brake_system):
    torque_ajustado = brake_system.apply_brake(pedal_force, coef_atrito)
    tempo_frenagem = calcular_tempo_frenagem_angular(brake_system, torque_ajustado)
    return tempo_frenagem

# Implementação do algoritmo de Evolução Diferencial Auto-adaptativa (EDA)
def evolucao_diferencial_adaptativa(func, brake_system, bounds, pop_size, F, CR, max_generations):
    dim = len(bounds)
    pop = np.random.rand(pop_size, dim)
    pop_denorm = bounds[:, 0] + pop * (bounds[:, 1] - bounds[:, 0])
    fitness = np.array([func(ind[0], ind[1], brake_system) for ind in pop_denorm])

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
            f = func(cruzado_denorm[0], cruzado_denorm[1], brake_system)
            if f < fitness[i]:
                fitness[i] = f
                pop[i] = cruzado
                if f < fitness[best_idx]:
                    best_idx = i
                    best = cruzado_denorm
        if abs(np.mean(fitness) - np.max(fitness)) < 1e-10:
            break
        gen += 1

    return best, fitness[best_idx]

# Definindo os parâmetros
params = {
    'RedP': 4, 'a': 0.8, 'psi': 0.48, 'μl': 0.45, 'pi': 3.14,
    'HCG': 0.5, 'μ': 1.5, 'FzF': 1471.5, 'FzR': 981.0, 'Rdp': 0.30,
    'Dcm': 0.02, 'Dwc': 0.032, 'Npast': 2, 'atrito_coeficiente': 0.35,
    'red': 0.75, 'Mt': 250, 'm_wheel': 10, 'm_tire': 10, 'L': 1.5, 'c_rr': 0.015
}
BrakeSystem = BrakeSystem(params)

# Definir os limites das variáveis de controle
bounds = np.array([[1, 823], [0.1, 0.45]])  # Força no pedal (N) e coeficiente de atrito

# Definir parâmetros de EDA
pop_size = 20
F = 0.8
CR = 0.9
max_generations = 100000
initial_speed = 20  # Velocidade inicial [m/s]
tempo = 2  # Tempo para o cálculo [s]

# Executar a EDA para otimizar o tempo de frenagem
melhor_solucao, melhor_tempo_frenagem = evolucao_diferencial_adaptativa(func_objetivo_frenagem_angular, BrakeSystem, bounds, pop_size, F, CR, max_generations)

print(f"Melhor Solução: Força no pedal = {melhor_solucao[0]} N, Coeficiente de atrito = {melhor_solucao[1]}")
print(f"Melhor Tempo de Frenagem: {melhor_tempo_frenagem} segundos")

# exemplo 2

import numpy as np

class BrakeSystem:
    # Inicializa a classe com os parâmetros necessários
    def __init__(self, params):
        self.params = params
        self.RedP = params['RedP']  # Redução do pedal
        self.a = params['a']  # Desaceleração [g]
        self.psi = params['psi']  # Distribuição de carga estática por eixo [adm]
        self.μl = params['μl']  # Coeficiente de atrito do contato pastilha/disco
        self.pi = params['pi']  # Valor de pi
        self.HCG = params['HCG']  # Altura do centro de gravidade [m]
        self.μ = params['μ']  # Coeficiente de atrito do pneu/solo
        self.FzF = params['FzF']  # Força de reação estática na dianteira [N]
        self.FzR = params['FzR']  # Força de reação estática na traseira [N]
        self.Rdp = params['Rdp']  # Raio do pneu [m]
        self.Dcm = params['Dcm']  # Diâmetro do cilindro mestre em metros
        self.Dwc = params['Dwc']  # Diâmetro do cilindro da roda em metros
        self.Npast = params['Npast']  # Número de pastilhas por disco
        self.atrito_coeficiente = params['atrito_coeficiente']  # Coeficiente de atrito da pastilha
        self.red = params['red']  # Raio efetivo do disco [m]
        self.Mt = params['Mt']  # Massa total do veículo [kg]
        self.L = params['L']  # Distância entre eixos [m]
        self.c_rr = params['c_rr']  # Coeficiente de resistência ao rolamento
        self.m_tire = params['m_tire']  # Massa do pneu
        self.m_wheel = params['m_wheel']  # Massa da roda

    # Calcula diversos parâmetros relacionados à frenagem
    def calculate_params(self, pedal_force=300):
        BF = 2 * self.μl  # Fator de freio
        χ = self.HCG / self.L  # Razão entre HCG e L
        W = self.Mt * 9.81  # Peso do veículo
        FzF_dyn = (1 - self.psi + self.a * χ) * W  # Força de reação dinâmica dianteira
        FzR_dyn = (self.psi - self.a * χ) * W  # Força de reação dinâmica traseira
        τF = FzF_dyn * self.μ * self.Rdp  # Torque na dianteira
        τR = FzR_dyn * self.μ * self.Rdp  # Torque na traseira
        FnF = τF / self.Npast * self.RedP * self.red  # Força normal das pastilhas dianteira
        FnR = τR / self.Npast * self.RedP * self.red  # Força normal das pastilhas traseira
        Awc = (self.pi * (self.Dwc ** 2)) / 4  # Área do cilindro de roda
        Acm = (self.pi * (self.Dcm ** 2)) / 4  # Área do cilindro mestre
        PF = FnF / Awc  # Pressão necessária na dianteira
        PR = FnR / Awc  # Pressão necessária na traseira
        FaCM = PF * Acm  # Força necessária para acionar o cilindro mestre
        return BF, FzF_dyn, FzR_dyn, τF, τR, FnF, FnR, Awc, Acm, PF, PR, FaCM

    # Aplica o freio e calcula a força de frenagem
    def apply_brake(self, pedal_force=300, coef_atrito=None):
        if coef_atrito:
            self.atrito_coeficiente = coef_atrito  # Atualiza o coeficiente de atrito

        resultados = self.calculate_params(pedal_force)
        BF, FzF_dyn, FzR_dyn, τF, τR, FnF, FnR, Awc, Acm, PF, PR, FaCM = resultados

        pressao_cilindro_mestre = pedal_force / Acm
        pressao_fluido = pressao_cilindro_mestre
        transmissao_pressao = pressao_fluido * Awc
        forca_aperto_pinca = transmissao_pressao
        forca_atrito_pastilha = forca_aperto_pinca * self.atrito_coeficiente
        torque_disco_freio = forca_atrito_pastilha * self.red
        resistencia_rolamento = self.c_rr * self.Mt * 9.81
        torque_resistencia_rolamento = resistencia_rolamento * self.Rdp
        torque_ajustado = torque_disco_freio - torque_resistencia_rolamento
        return torque_ajustado

    def calculate_angular_velocity(self, torque_ajustado, initial_speed, tempo):
        time_intervals = np.linspace(0, tempo, 100)  # 100 intervalos de tempo entre 0 e o tempo total
        inertia_wheel = 0.5 * (self.m_tire + self.m_wheel) * (self.Rdp ** 2)  # Inércia da roda
        angular_desaceleration = torque_ajustado / inertia_wheel  # Desaceleração angular
        initial_angular_velocity = initial_speed / self.Rdp  # Velocidade angular inicial
        angular_velocity = initial_angular_velocity
        angular_velocities = []

        for t in time_intervals:
            angular_velocity = initial_angular_velocity - angular_desaceleration * t
            if angular_velocity < 0:
                angular_velocity = 0
            angular_velocities.append(angular_velocity)

        return angular_desaceleration, angular_velocities



def calcular_tempo_frenagem_angular(brake_system, torque_ajustado, initial_speed, tempo):
    angular_desaceleration, angular_velocities = brake_system.calculate_angular_velocity(torque_ajustado, initial_speed, tempo)
    
    # Encontrar o tempo em que a velocidade angular chega a zero
    tempo_frenagem = np.linspace(0, tempo, len(angular_velocities))
    velocidades_array = np.array(angular_velocities)
    
    # Encontrar o primeiro índice onde a velocidade é zero ou negativa
    indices = np.where(velocidades_array <= 0)[0]
    
    if len(indices) > 0:
        tempo_frenagem = tempo_frenagem[indices[0]]
    else:
        # Se a velocidade angular nunca chega a zero, considere o tempo máximo como resultado
        tempo_frenagem = np.inf

    return tempo_frenagem


def func_objetivo_frenagem_angular(pedal_force, coef_atrito, brake_system, initial_speed, tempo):
    torque_ajustado = brake_system.apply_brake(pedal_force, coef_atrito)
    tempo_frenagem = calcular_tempo_frenagem_angular(brake_system, torque_ajustado, initial_speed, tempo)
    return tempo_frenagem


def evolucao_diferencial_adaptativa(func, brake_system, bounds, pop_size, F, CR, max_generations, initial_speed, tempo):
    dim = len(bounds)
    pop = np.random.rand(pop_size, dim)
    pop_denorm = bounds[:, 0] + pop * (bounds[:, 1] - bounds[:, 0])
    fitness = np.array([func(ind[0], ind[1], brake_system, initial_speed, tempo) for ind in pop_denorm])

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
            f = func(cruzado_denorm[0], cruzado_denorm[1], brake_system, initial_speed, tempo)
            if f < fitness[i]:
                fitness[i] = f
                pop[i] = cruzado
                if f < fitness[best_idx]:
                    best_idx = i
                    best = cruzado_denorm
        if abs(np.mean(fitness) - np.max(fitness)) < 1e-10:
            break
        gen += 1

    return best, fitness[best_idx]

# Definindo os parâmetros
params = {
    'RedP': 4, 'a': 0.8, 'psi': 0.48, 'μl': 0.45, 'pi': 3.14,
    'HCG': 0.5, 'μ': 1.5, 'FzF': 1471.5, 'FzR': 981.0, 'Rdp': 0.30,
    'Dcm': 0.02, 'Dwc': 0.032, 'Npast': 2, 'atrito_coeficiente': 0.35,
    'red': 0.75, 'Mt': 250, 'm_wheel': 10, 'm_tire': 10, 'L': 1.5, 'c_rr': 0.015
}
BrakeSystem = BrakeSystem(params)

# Definir os limites das variáveis de controle
bounds = np.array([[1, 823], [0.1, 0.45]])  # Força no pedal (N) e coeficiente de atrito

# Definir parâmetros de EDA
pop_size = 20
F = 0.8
CR = 0.9
max_generations = 100000
initial_speed = 20  # Velocidade inicial [m/s]
tempo = 2  # Tempo para o cálculo [s]

# Executar a EDA para otimizar o tempo de frenagem
melhor_solucao, melhor_tempo_frenagem = evolucao_diferencial_adaptativa(
    func_objetivo_frenagem_angular,
    BrakeSystem,
    bounds,
    pop_size,
    F,
    CR,
    max_generations,
    initial_speed,
    tempo
)

print(f"Melhor Solução: Força no pedal = {melhor_solucao[0]} N, Coeficiente de atrito = {melhor_solucao[1]}")
print(f"Melhor Tempo de Frenagem: {melhor_tempo_frenagem} segundos")


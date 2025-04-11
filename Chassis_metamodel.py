import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, solve_ivp
import functools

# =============================================================================
# 1. Modelo e Métodos do Chassis
# =============================================================================

class Chassis:
    def __init__(self, 
                Mc=None, Mst=None, Msd=None,                   # Massas
                Kte=None, Ktd=None, Kde=None, Kdd=None,        # Rigidez suspensão
                Kpte=None, Kptd=None, Kpde=None, Kpdd=None,    # Rigidez pneus
                Kf=None, Kt=None,                              # Rigidez do chassi
                Cte=None, Ctd=None, Cde=None, Cdd=None,        # Amortecedores suspensão
                Cphi=None, Ctheta=None,                        # Amortecedores do chassi
                Iflex=None, Itorc=None):                       # Inércias do chassi
        # Massas
        self.Mc = Mc                    # Massa suspensa TOTAL [kg]
        self.Mst_unsprung = Mst  # Massa NÃO suspensa traseira [kg]
        self.Msd_unsprung = Msd  # Massa NÃO suspensa dianteira [kg]
        
        # Rigidez da suspensão
        self.Kte = Kte    # Traseira esquerda [N/m]
        self.Ktd = Ktd    # Traseira direita [N/m]
        self.Kde = Kde    # Dianteira esquerda [N/m]
        self.Kdd = Kdd    # Dianteira direita [N/m]
        
        # Rigidez dos pneus
        self.Kpte = Kpte  # Pneu traseiro esquerdo [N/m]
        self.Kptd = Kptd  # Pneu traseiro direito [N/m]
        self.Kpde = Kpde  # Pneu dianteiro esquerdo [N/m]
        self.Kpdd = Kpdd  # Pneu dianteiro direito [N/m]
        
        # Rigidez do chassi
        self.Kf = Kf      # Flexional [N/m]
        self.Kt = Kt      # Rigidez do pneu (ou transferência) [N/m]
        
        # Amortecedores da suspensão
        self.Cte = Cte    # Traseira esquerda [N·s/m]
        self.Ctd = Ctd    # Traseira direita [N·s/m]
        self.Cde = Cde    # Dianteira esquerda [N·s/m]
        self.Cdd = Cdd    # Dianteira direita [N·s/m]
        
        # Amortecedores estruturais do chassi
        self.Cphi = Cphi      # Torcional [N·s/m]
        self.Ctheta = Ctheta  # Flexional [N·s/m]
        
        # Inércias do chassi
        self.Iflex = Iflex    # Inércia flexional [m⁴]
        self.Itorc = Itorc    # Inércia torcional [m⁴]

    def spring(self, mass, k, x):
        """
        Calcula aceleração devido à mola.
        Retorna: (k * x) / mass
        """
        return (k * x) / mass

    def damper(self, mass, c, xdot):
        """
        Calcula aceleração devido ao amortecedor.
        Retorna: (c * xdot) / mass
        """
        return (c * xdot) / mass

    def integrate_motion(self, t, acceleration):
        """
        Integra a aceleração para obter velocidade e deslocamento.
        t: vetor de tempo
        acceleration: vetor de aceleração
        Retorna: (velocity, displacement) com condições iniciais zero.
        """
        velocity = cumulative_trapezoid(acceleration, t, initial=0)
        displacement = cumulative_trapezoid(velocity, t, initial=0)
        return velocity, displacement

    def ode_wheel(self, t, y, mass, Ks, Cs, Kt, z_road_func):
        """
        Equação de segunda ordem da roda:
            m * x'' = -Ks * x - Cs * x' + Kt * z_r(t)
        Utiliza os métodos spring e damper para calcular as acelerações individuais.
        """
        x, xdot = y
        zr = z_road_func(t)
        # Aceleração da mola e do amortecedor (dividida por mass)
        spring_acc = self.spring(mass, Ks, x)   # (Ks*x)/mass
        damper_acc = self.damper(mass, Cs, xdot)  # (Cs*xdot)/mass
        entrance_acc = self.spring(mass, Kt, x=zr)
        # Equação de movimento: x'' = -spring - damper + (Kt/mass)*zr
        xddot = -spring_acc - damper_acc + entrance_acc
        return [xdot, xddot]

# =============================================================================
# 2. Funções de Entrada e Configuração
# =============================================================================

def gerar_funcoes_entrada(tempo_direito, tempo_esquerdo, step_value_d, step_value_e):
    """
    Gera funções step para as rodas dianteiras e traseiras.
    Cada função retorna 'altura' se t >= t_start, ou 0 caso contrário.
    """
    def road_input_step(t, t_start, height):
        return height if t >= t_start else 0.0

    func_direito = functools.partial(road_input_step, t_start=tempo_direito, height=step_value_d)
    func_esquerdo = functools.partial(road_input_step, t_start=tempo_esquerdo, height=step_value_e)
    return {'FL': func_esquerdo, 'FR': func_direito,
            'RL': func_esquerdo, 'RR': func_direito}

def get_simulation_time(t_start=0.0, t_end=2.0, dt=0.001):
    """Retorna t_span e vetor de avaliação t_eval."""
    t_span = (t_start, t_end)
    t_eval = np.arange(t_start, t_end+dt, dt)
    return t_span, t_eval

# =============================================================================
# 3. Rotina de Simulação e Plotagem
# =============================================================================

def simular_roda(chassis, mass, Ks, Cs, Kt, t_span, t_eval, z_road_func):
    """Resolve a ODE para uma roda isolada."""
    y0 = [0.0, 0.0]  # condições iniciais: deslocamento 0 e velocidade 0
    sol = solve_ivp(fun=lambda t, y: chassis.ode_wheel(t, y, mass, Ks, Cs, Kt, z_road_func),
                    t_span=t_span, y0=y0, t_eval=t_eval,
                    method='LSODA', rtol=1e-6, atol=1e-9)
    return sol

def plotar_resultados(rodas_solucoes, titulo='Resposta ao degrau'):
    """Plota os deslocamentos (vertical) para cada roda."""
    nomes = {'FL': 'Dianteira Esquerda', 'FR': 'Dianteira Direita',
            'RL': 'Traseira Esquerda', 'RR': 'Traseira Direita'}
    cores = {'FL': 'purple', 'FR': 'blue', 'RL': 'green', 'RR': 'red'}
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True, constrained_layout=True)
    fig.suptitle(titulo, fontsize=14)
    
    for i, key in enumerate(['FL', 'FR', 'RL', 'RR']):
        sol = rodas_solucoes[key]
        axs[i].plot(sol.t, sol.y[0], label=nomes[key], color=cores[key])
        axs[i].set_ylabel('Deslocamento (m)')
        axs[i].legend(loc='upper right')
        axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        axs[i].axhline(0, color='black', linewidth=0.5)
    
    axs[-1].set_xlabel('Tempo (s)')
    plt.show()

def run_simulation():
    # -------------------------------
    # Parâmetros de rodas e suspensão
    Mst_unsprung = 25.0  # para traseira
    Msd_unsprung = 20.0  # para dianteira
    Ks_susp = 5e5       # Rigidez da suspensão [N/m]
    Cs_front = 3e3      # Amortecimento dianteiro [N·s/m]
    Cs_rear = 5e3       # Amortecimento traseiro [N·s/m]
    Kt_tire = 2e4       # Rigidez do pneu [N/m]
    
    # -------------------------------
    # Configuração da simulação temporal
    t_span, t_eval = get_simulation_time(t_start=0.0, t_end=6.0, dt=0.001)
    
    # Configuração do degrau e atraso: as rodas traseiras recebem o degrau com atraso
    Ld, Lt = 1.0, 0.8         # distância de referência para o atraso
    velocidade = 5.0          # velocidade do veículo (m/s)
    altura_degrau_d = 0.10
    altura_degrau_e = 0.20
    tempo_inicio_direito = 1.0
    tempo_inicio_esquerdo = 2.0
    
    # Gera funções de entrada para as quatro rodas
    funcoes_entrada = gerar_funcoes_entrada(tempo_inicio_direito, tempo_inicio_esquerdo, altura_degrau_d, altura_degrau_e)
    
    # -------------------------------
    # Instanciando o modelo (Chassis)
    chassis = Chassis(
        Mc=280,
        Mst=Mst_unsprung,
        Msd=Msd_unsprung,
        Kte=5e5, Ktd=5e5, Kde=5e5, Kdd=5e5,
        Kpte=2e4, Kptd=2e4, Kpde=2e4, Kpdd=2e4,
        Kf=3.01e7, Kt=Kt_tire,
        Cte=5e3, Ctd=5e3, Cde=3e3, Cdd=3e3,
        Cphi=5e2, Ctheta=5e2,
        Iflex=5e5, Itorc=5e5
    )
    
    # -------------------------------
    # Simular cada roda usando os parâmetros e funções de entrada apropriadas
    solucoes = {}
    # Para rodas dianteiras usamos os mesmos parâmetros para ambas
    for roda in ['FL', 'FR']:
        solucoes[roda] = simular_roda(chassis, Msd_unsprung, Ks_susp, Cs_front, Kt_tire,
                                    t_span, t_eval, funcoes_entrada[roda])
    # Para rodas traseiras
    for roda in ['RL', 'RR']:
        solucoes[roda] = simular_roda(chassis, Mst_unsprung, Ks_susp, Cs_rear, Kt_tire,
                                    t_span, t_eval, funcoes_entrada[roda])
    
    # Plotar os resultados
    plotar_resultados(solucoes, titulo='Simulação de Resposta de Rodas Isoladas (Degrau)')
    
    # Exemplo opcional: Demonstrar a integração de aceleração para uma roda (por exemplo, FL)
    xdot_FL = np.gradient(solucoes['FL'].y[0], solucoes['FL'].t)
    xddot_FL = np.gradient(xdot_FL, solucoes['FL'].t)
    vel_int, disp_int = chassis.integrate_motion(solucoes['FL'].t, xddot_FL)
    plt.figure(figsize=(10, 4))
    plt.plot(solucoes['FL'].t, xdot_FL, label='Velocidade (gradiente)')
    plt.plot(solucoes['FL'].t, vel_int, '--', label='Velocidade (integrada)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Velocidade (m/s)')
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================================================
# Execução Principal
# =============================================================================

run_simulation()

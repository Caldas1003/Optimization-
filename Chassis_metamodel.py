import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, solve_ivp
import functools

# ---------------------- Classe Chassis (mantida como está) ----------------------

class Chassis:
    def __init__(self, 
                Mc=None, Mst=None, Msd=None,                   # Massas
                Kte=None, Ktd=None, Kde=None, Kdd=None,        # Rigidez suspensão
                Kpte=None, Kptd=None, Kpde=None, Kpdd=None,    # Rigidez pneus
                Kf=None, Kt=None,                              # Rigidez do chassi
                Cte=None, Ctd=None, Cde=None, Cdd=None,        # Amortecedores suspensão
                Cphi=None, Ctheta=None,                        # Amortecedores do chassi
                Iflex=None, Itorc=None,                       # Inércias do chassi
                Mc_sprung=None, Mst_unsprung=None, Msd_unsprung=None):  # Massa suspensa TOTAL, Massa não suspensa traseira e Massa não suspensa dianteira
        
        # Massas
        self.Mc = Mc            # Massa do carro [kg]
        self.Mst = Mst          # Massa suspensa traseira [kg]
        self.Msd = Msd          # Massa suspensa dianteira [kg]
        self.Mc = Mc_sprung     # Massa suspensa TOTAL [kg]
        self.Mst = Mst_unsprung # Massa NÃO suspensa traseira [kg]
        self.Msd = Msd_unsprung # Massa NÃO suspensa dianteira [kg]

        # Rigidez da suspensão
        self.Kte = Kte      # Traseira esquerda [N/m]
        self.Ktd = Ktd      # Traseira direita [N/m]
        self.Kde = Kde      # Dianteira esquerda [N/m]
        self.Kdd = Kdd      # Dianteira direita [N/m]

        # Rigidez dos pneus
        self.Kpte = Kpte    # Pneu traseiro esquerdo [N/m]
        self.Kptd = Kptd    # Pneu traseiro direito [N/m]
        self.Kpde = Kpde    # Pneu dianteiro esquerdo [N/m]
        self.Kpdd = Kpdd    # Pneu dianteiro direito [N/m]

        # Rigidez do chassi
        self.Kf = Kf        # Flexional [N/m]
        self.Kt = Kt        # Torcional [N/m]

        # Amortecedores da suspensão
        self.Cte = Cte      # Traseira esquerda [N·s/m]
        self.Ctd = Ctd      # Traseira direita [N·s/m]
        self.Cde = Cde      # Dianteira esquerda [N·s/m]
        self.Cdd = Cdd      # Dianteira direita [N·s/m]

        # Amortecedores estruturais do chassi
        self.Cphi = Cphi      # Torcional [N·s/m]
        self.Ctheta = Ctheta  # Flexional [N·s/m]

        # Inércias do chassi
        self.If_ = Iflex    # Inércia flexional [m⁴]
        self.It = Itorc     # Inércia torcional [m⁴]

    def spring(self, mass, k, x):
        F = k * x
        acceleration = F / mass
        return acceleration

    def damper(self, mass, c, xdot):
        F = c * xdot
        acceleration = F / mass
        return acceleration

    def integrate_motion(t, acceleration):
        """
        Integra a aceleração para obter velocidade e deslocamento.
        t: vetor de tempo
        acceleration: vetor de aceleração
        Retorna (velocity, displacement) com condições iniciais zero.
        """
        velocity = cumulative_trapezoid(acceleration, t, initial=0)
        displacement = cumulative_trapezoid(velocity, t, initial=0)
        return velocity, displacement


def gerar_funcoes_entrada(tempo_dianteira, tempo_traseira, altura):
    def road_input_step(t, t_start, height):
        return height if t >= t_start else 0.0

    func_dianteira = functools.partial(road_input_step, t_start=tempo_dianteira, height=altura)
    func_traseira = functools.partial(road_input_step, t_start=tempo_traseira, height=altura)

    return {
        'FL': func_dianteira,
        'FR': func_dianteira,
        'RL': func_traseira,
        'RR': func_traseira
    }

def edo_roda_isolada(t, y, Mu, Ks, Cs, Kt, z_road_func):
    xu, xu_dot = y
    zr = z_road_func(t)
    xu_ddot = (Kt * zr - Cs * xu_dot - (Ks + Kt) * xu) / Mu
    return [xu_dot, xu_ddot]

def simular_rodas_isoladas(params_map, funcoes_entrada, t_span, t_eval):
    resultados = {}
    for canto, params in params_map.items():
        Mu, Ks, Cs, Kt = params['Mu'], params['Ks'], params['Cs'], params['Kt']
        y0 = [0, 0]
        sol = solve_ivp(
            fun=edo_roda_isolada, t_span=t_span, y0=y0, t_eval=t_eval,
            args=(Mu, Ks, Cs, Kt, funcoes_entrada[canto]),
            method='LSODA', rtol=1e-6, atol=1e-9
        )
        resultados[canto] = sol
    return resultados

def plotar_resultados(resultados, params_map, Ks):
    nomes_plot = {
        'FR': 'Xdd (roda dianteira direita)',
        'RR': 'Xtd (roda traseira direita)',
        'RL': 'Xte (roda traseira esquerda)',
        'FL': 'Xde (roda dianteira esquerda)'
    }
    ordem = ['FR', 'RR', 'RL', 'FL']
    cores = ['blue', 'red', 'green', 'purple']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True, constrained_layout=True)
    fig.suptitle(f'Simulação Roda Isolada (Ks={Ks:.0e})', fontsize=14)

    for i, canto in enumerate(ordem):
        ax = axs[i]
        sol = resultados[canto]
        ax.plot(sol.t, sol.y[0], label=nomes_plot[canto], color=cores[i])

        params = params_map[canto]
        Mu, Ks, Cs, Kt = params['Mu'], params['Ks'], params['Cs'], params['Kt']
        omega_n = np.sqrt((Ks + Kt) / Mu)
        Cc = 2 * Mu * omega_n
        zeta = Cs / Cc if Cc > 0 else 0

        ax.set_title(f'{nomes_plot[canto]} (ζ ≈ {zeta:.3f})')
        ax.set_ylabel('Deslocamento (m)')
        ax.legend(loc='lower right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.5)

    axs[-1].set_xlabel('Tempo (s)')
    plt.show()

def run_simulation():
    # Parâmetros do sistema
    Mst_unsprung = 25.0
    Msd_unsprung = 20.0
    Ks_susp = 5e5
    Cs_front = 3e3
    Cs_rear = 5e3
    Kt_tire = 2e4

    params_map = {
        'FL': {'Mu': Msd_unsprung, 'Ks': Ks_susp, 'Cs': Cs_front, 'Kt': Kt_tire},
        'FR': {'Mu': Msd_unsprung, 'Ks': Ks_susp, 'Cs': Cs_front, 'Kt': Kt_tire},
        'RL': {'Mu': Mst_unsprung, 'Ks': Ks_susp, 'Cs': Cs_rear,  'Kt': Kt_tire},
        'RR': {'Mu': Mst_unsprung, 'Ks': Ks_susp, 'Cs': Cs_rear,  'Kt': Kt_tire}
    }

    # Configurações da simulação
    t_start, t_end, dt = 0.0, 2.0, 0.001
    t_span = (t_start, t_end)
    t_eval = np.arange(t_start, t_end + dt, dt)

    # Configuração do degrau e atraso traseiro
    Ld, Lt = 1.0, 0.8
    velocidade = 5.0
    atraso_traseira = (Ld + Lt) / velocidade
    altura_degrau = 0.10
    tempo_inicio_dianteira = 1.0
    tempo_inicio_traseira = tempo_inicio_dianteira + atraso_traseira

    # Gerar funções de entrada
    funcoes_entrada = gerar_funcoes_entrada(tempo_inicio_dianteira, tempo_inicio_traseira, altura_degrau)

    # Rodar simulação e plotar resultados
    resultados = simular_rodas_isoladas(params_map, funcoes_entrada, t_span, t_eval)
    plotar_resultados(resultados, params_map, Ks_susp)


run_simulation()

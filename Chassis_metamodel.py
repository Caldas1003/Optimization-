
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, solve_ivp
import functools

# =============================================================================
# Classe do Chassis (sem alterações)
# =============================================================================

class Chassis:
    def __init__(self, 
                Mc=None, Mst=None, Msd=None,              # Massas
                Kte=None, Ktd=None, Kde=None, Kdd=None,  # Rigidez suspensão
                Kpte=None, Kptd=None, Kpde=None, Kpdd=None,  # Rigidez pneus
                Kf=None, Kt=None,                            # Rigidez do chassi
                Cte=None, Ctd=None, Cde=None, Cdd=None,  # Amortecedores suspensão
                Cphi=None, Ctheta=None,                       # Amortecedores do chassi
                Iflex=None, Itorc=None):                       # Inércias do chassi
        # Massas
        self.Mc = Mc            # Massa suspensa TOTAL [kg]
        self.Mst_unsprung = Mst   # Massa NÃO suspensa traseira [kg]
        self.Msd_unsprung = Msd   # Massa NÃO suspensa dianteira [kg]
        
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
        return (k * x) / mass

    def damper(self, mass, c, xdot):
        return (c * xdot) / mass
    
    def integrate_motion(self, t, acceleration):
        velocity = cumulative_trapezoid(acceleration, t, initial=0)
        displacement = cumulative_trapezoid(velocity, t, initial=0)
        return velocity, displacement
    
    def gerar_funcoes_entrada(self, tempo_direito, tempo_esquerdo, step_value_d, step_value_e):
        def road_input_step(t, t_start, height):
            return height if t >= t_start else 0.0
        func_direito = functools.partial(road_input_step, t_start=tempo_direito, height=step_value_d)
        func_esquerdo = functools.partial(road_input_step, t_start=tempo_esquerdo, height=step_value_e)
        return {'FL': func_esquerdo, 'FR': func_direito,
                'RL': func_esquerdo, 'RR': func_direito}
    
    def ode_wheel(self, t, y, mass, Ks, Cs, Kt, z_road_func):
        x, xdot = y
        zr = z_road_func(t)
        spring_acc = self.spring(mass, Ks, x)
        damper_acc = self.damper(mass, Cs, xdot)
        entrance_acc = self.spring(mass, Kt, zr)
        xddot = -spring_acc - damper_acc + entrance_acc
        return [xdot, xddot]

    def simular_roda(self, roda, t_span, t_eval, z_road_func):
        if roda in ['FL', 'FR']:
            mass = self.Msd_unsprung
            Ks = self.Kde if roda == 'FL' else self.Kdd
            Cs = self.Cde if roda == 'FL' else self.Cdd
            Kt = self.Kpde if roda == 'FL' else self.Kpdd
        else:
            mass = self.Mst_unsprung
            Ks = self.Kte if roda == 'RL' else self.Ktd
            Cs = self.Cte if roda == 'RL' else self.Ctd
            Kt = self.Kpte if roda == 'RL' else self.Kptd

        y0 = [0.0, 0.0]
        sol = solve_ivp(fun=lambda t, y: self.ode_wheel(t, y, mass, Ks, Cs, Kt, z_road_func),
                        t_span=t_span, y0=y0, t_eval=t_eval,
                        method='LSODA', rtol=1e-6, atol=1e-9)
        return sol

    def ode_theta(self, solucoes, t_eval):
        """
        Calcula a vibração do chassi: heave, pitch (flexão) e roll (torção).
        """
        print("\nAnalisando a vibração total do chassi (heave, pitch, roll)...")

        # 1. Extrair os deslocamentos (x) e velocidades (x_dot) de cada roda
        x_fl, xdot_fl = solucoes['FL'].y
        x_fr, xdot_fr = solucoes['FR'].y
        x_rl, xdot_rl = solucoes['RL'].y
        x_rr, xdot_rr = solucoes['RR'].y
        
        # 2. Calcular forças nas suspensões
        acc_FL = self.spring(self.Mc, self.Kde, x_fl) + self.damper(self.Mc, self.Cde, xdot_fl)
        acc_FR = self.spring(self.Mc, self.Kdd, x_fr) + self.damper(self.Mc, self.Cdd, xdot_fr)
        acc_RL = self.spring(self.Mc, self.Kte, x_rl) + self.damper(self.Mc, self.Cte, xdot_rl)
        acc_RR = self.spring(self.Mc, self.Ktd, x_rr) + self.damper(self.Mc, self.Ctd, xdot_rr)

        # 3. Força total vertical (heave)
        acc_total = acc_FL + acc_FR + acc_RL + acc_RR
        heave = self.integrate_motion(t_eval, acc_total)[1]
        return heave
    
    def ode_phi(self, solucoes, t_eval):
        """
        Calcula a vibração do chassi: heave, pitch (flexão) e roll (torção).
        """
        print("\nAnalisando a vibração total do chassi (heave, pitch, roll)...")

        # 1. Extrair os deslocamentos (x) e velocidades (x_dot) de cada roda
        x_fl, xdot_fl = solucoes['FL'].y
        x_fr, xdot_fr = solucoes['FR'].y
        x_rl, xdot_rl = solucoes['RL'].y
        x_rr, xdot_rr = solucoes['RR'].y
        
        # 2. Calcular forças nas suspensões
        acc_FL = self.spring(self.Mc, self.Kde, x_fl) + self.damper(self.Mc, self.Cde, xdot_fl)
        acc_FR = self.spring(self.Mc, self.Kdd, x_fr) + self.damper(self.Mc, self.Cdd, xdot_fr)
        acc_RL = self.spring(self.Mc, self.Kte, x_rl) + self.damper(self.Mc, self.Cte, xdot_rl)
        acc_RR = self.spring(self.Mc, self.Ktd, x_rr) + self.damper(self.Mc, self.Ctd, xdot_rr)

        # 3. Força total vertical (heave)
        acc_total = acc_FL + acc_FR + acc_RL + acc_RR
        heave = self.integrate_motion(t_eval, acc_total)[1]
        return heave

    def plotar_resultados_rodas(self, rodas_solucoes, titulo='Resposta ao degrau'):
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

    def plotar_vibracao_chassi(t_eval, posicao_vertical):
        """
        Plota o gráfico com a posição vertical do chassi.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(t_eval, posicao_vertical, label='Posição Vertical do Chassi (Heave)', color='navy')
        plt.title('Posição Vertical do Chassi Integrada da Aceleração Total', fontsize=16)
        plt.xlabel('Tempo (s)', fontsize=12)
        plt.ylabel('Deslocamento Vertical (m)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.legend()
        plt.show()

# =============================================================================
# Função para obter vetor de tempo
# =============================================================================
def get_simulation_time(t_start=0.0, t_end=2.0, dt=0.001):
    t_span = (t_start, t_end)
    t_eval = np.arange(t_start, t_end+dt, dt)
    return t_span, t_eval

# =============================================================================
# Execução Principal 
# =============================================================================
def run_simulation():
    # 1. Definição de Parâmetros
    Mst_unsprung = 25.0
    Msd_unsprung = 20.0
    Ks_susp = 5e5
    Cs_front = 3e3
    Cs_rear = 5e3
    Kt_tire = 2e4
    Mc_chassi = 280.0
    
    t_span, t_eval = get_simulation_time(t_start=0.0, t_end=30.0, dt=0.001)
    
    altura_degrau_d = 0.10
    altura_degrau_e = 0.20
    tempo_inicio_direito = 1.0
    tempo_inicio_esquerdo = 2.0

    # 2. Criação dos objetos
    chassis = Chassis(
        Mc=Mc_chassi, Mst=Mst_unsprung, Msd=Msd_unsprung,
        Kte=Ks_susp, Ktd=Ks_susp, Kde=Ks_susp, Kdd=Ks_susp,
        Kpte=Kt_tire, Kptd=Kt_tire, Kpde=Kt_tire, Kpdd=Kt_tire,
        Kf=3.01e7, Kt=Kt_tire,
        Cte=Cs_rear, Ctd=Cs_rear, Cde=Cs_front, Cdd=Cs_front,
        Cphi=5e2, Ctheta=5e2,
        Iflex=5e5, Itorc=5e5
    )

    funcoes_entrada = chassis.gerar_funcoes_entrada(tempo_inicio_direito, tempo_inicio_esquerdo,
                                                    altura_degrau_d, altura_degrau_e)

    # 3. Simulação das rodas
    solucoes = {}
    print("Simulando as rodas independentemente...")
    for roda in ['FL', 'FR', 'RL', 'RR']:
        solucoes[roda] = chassis.simular_roda(roda, t_span, t_eval, funcoes_entrada[roda])
    
    # 4. Plotagem dos resultados das rodas
    chassis.plotar_resultados_rodas(solucoes, titulo='Simulação de Resposta de Rodas Isoladas (Degrau)')

    # 5. Análise da vibração do chassi 
    posicao_chassi = chassis.analisar_vibracao_chassi(solucoes, t_eval)[0]
    
    # 6. Plotagem da vibração do chassi 
    chassis.plotar_vibracao_chassi(t_eval, posicao_chassi)

# =============================================================================
# Rodar a simulação
# =============================================================================

run_simulation()

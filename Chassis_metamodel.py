import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, solve_ivp
import functools

# =============================================================================
# Modelo do Chassis com todas as funções internas organizadas
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
        return (k * x) / mass

    def damper(self, mass, c, xdot):
        return (c * xdot) / mass

    def integrate_motion(self, t, acceleration):
        velocity = cumulative_trapezoid(acceleration, t, initial=0)
        displacement = cumulative_trapezoid(velocity, t, initial=0)
        return velocity, displacement
    
    def ode_phi(self, y, inertia, roda):
        #Rigidez e amortecimeto torsional do Chassis
        Ks = self.Kt
        Cs = self.Cphi
        inertia = self.Itorc
        phi, phidot = y
        spring_acc = self.spring(inertia, Ks, phi)
        damper_acc = self.damper(inertia, Cs, phidot)
        wheel_acc = 0

        for roda in ['FL', 'FR', 'RL', 'RR']:
            if roda in ['FL', 'FR']:
                Ks = self.Kde if roda == 'FL' else self.Kdd
                Cs = self.Cde if roda == 'FL' else self.Cdd
                Kt = self.Kpde if roda == 'FL' else self.Kpdd
            else:
                Ks = self.Kte if roda == 'RL' else self.Ktd
                Cs = self.Cte if roda == 'RL' else self.Ctd
                Kt = self.Kpte if roda == 'RL' else self.Kptd

            acc = self.spring(inertia, Ks, phi) + self.damper(inertia, Cs, phidot)


        phiddot = -spring_acc - damper_acc
        return [phidot, phiddot]
    
    def ode_theta(self, y, inertia):
        Ks = self.Kf
        Cs = self.Ctheta
        inertia = self.Iflex
        theta, thetadot = y
        spring_acc = self.spring(inertia, Ks, theta)
        damper_acc = self.damper(inertia, Cs, thetadot)
        thetaddot = -spring_acc - damper_acc
        return [thetadot, thetaddot]

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

    def gerar_funcoes_entrada(self, tempo_direito, tempo_esquerdo, step_value_d, step_value_e):
        def road_input_step(t, t_start, height):
            return height if t >= t_start else 0.0
        func_direito = functools.partial(road_input_step, t_start=tempo_direito, height=step_value_d)
        func_esquerdo = functools.partial(road_input_step, t_start=tempo_esquerdo, height=step_value_e)
        return {'FL': func_esquerdo, 'FR': func_direito,
                'RL': func_esquerdo, 'RR': func_direito}

    def plotar_resultados(self, rodas_solucoes, titulo='Resposta ao degrau'):
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

    def demonstrar_integracao(self, sol):
        xdot = np.gradient(sol.y[0], sol.t)
        xddot = np.gradient(xdot, sol.t)
        vel_int, _ = self.integrate_motion(sol.t, xddot)
        plt.figure(figsize=(10, 4))
        plt.plot(sol.t, xdot, label='Velocidade (gradiente)')
        plt.plot(sol.t, vel_int, '--', label='Velocidade (integrada)')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Velocidade (m/s)')
        plt.legend()
        plt.grid(True)
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
    # Parâmetros
    Mst_unsprung = 25.0
    Msd_unsprung = 20.0
    Ks_susp = 5e5
    Cs_front = 3e3
    Cs_rear = 5e3
    Kt_tire = 2e4
    
    t_span, t_eval = get_simulation_time(t_start=0.0, t_end=6.0, dt=0.001)
    
    # Degraus
    altura_degrau_d = 0.10
    altura_degrau_e = 0.20
    tempo_inicio_direito = 1.0
    tempo_inicio_esquerdo = 2.0

    # Chassis
    chassis = Chassis(
        Mc=280, Mst=Mst_unsprung, Msd=Msd_unsprung,
        Kte=Ks_susp, Ktd=Ks_susp, Kde=Ks_susp, Kdd=Ks_susp,
        Kpte=Kt_tire, Kptd=Kt_tire, Kpde=Kt_tire, Kpdd=Kt_tire,
        Kf=3.01e7, Kt=Kt_tire,
        Cte=Cs_rear, Ctd=Cs_rear, Cde=Cs_front, Cdd=Cs_front,
        Cphi=5e2, Ctheta=5e2,
        Iflex=5e5, Itorc=5e5
    )

    funcoes_entrada = chassis.gerar_funcoes_entrada(tempo_inicio_direito, tempo_inicio_esquerdo,
                                                    altura_degrau_d, altura_degrau_e)

    solucoes = {}
    for roda in ['FL', 'FR', 'RL', 'RR']:
        solucoes[roda] = chassis.simular_roda(roda, t_span, t_eval, funcoes_entrada[roda])
    print(solucoes)
    chassis.plotar_resultados(solucoes, titulo='Simulação de Resposta de Rodas Isoladas (Degrau)')
    chassis.demonstrar_integracao(solucoes['FL'])

# =============================================================================
# Rodar a simulação
# =============================================================================

run_simulation()

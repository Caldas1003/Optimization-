import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import functools

# =============================================================================
# Modelo do Chassis com todas as funções internas organizadas
# =============================================================================

class Chassis:
    def __init__(self, 
                Mc=None, Mst=None, Msd=None,                  # Massas
                Kte=None, Ktd=None, Kde=None, Kdd=None,       # Rigidez suspensão
                Kpte=None, Kptd=None, Kpde=None, Kpdd=None,   # Rigidez pneus
                Kf=None, Kt=None,                             # Rigidez do chassi
                Cte=None, Ctd=None, Cde=None, Cdd=None,       # Amortecedores suspensão
                Cphi=None, Ctheta=None,                       # Amortecedores do chassi
                Iflex=None, Itorc=None,                       # Inércias do chassi
                W=None, Lt=None, Ld=None):                    # Unidade de comprimento
        
        # Massas
        self.Mc = Mc               # Massa suspensa TOTAL [kg]
        self.Mst_unsprung = Mst    # Massa NÃO suspensa traseira [kg]
        self.Msd_unsprung = Msd    # Massa NÃO suspensa dianteira [kg]
        
        # Rigidez da suspensão
        self.Kte = Kte   # Traseira esquerda [N/m]
        self.Ktd = Ktd   # Traseira direita [N/m]
        self.Kde = Kde   # Dianteira esquerda [N/m]
        self.Kdd = Kdd   # Dianteira direita [N/m]
        
        # Rigidez dos pneus
        self.Kpte = Kpte # Pneu traseiro esquerdo [N/m]
        self.Kptd = Kptd # Pneu traseiro direito [N/m]
        self.Kpde = Kpde # Pneu dianteiro esquerdo [N/m]
        self.Kpdd = Kpdd # Pneu dianteiro direito [N/m]
        
        # Rigidez do chassi
        self.Kf = Kf     # Flexional [N/m]
        self.Kt = Kt     # Torcional [N/m]
        
        # Amortecedores da suspensão
        self.Cte = Cte   # Traseira esquerda [N·s/m]
        self.Ctd = Ctd   # Traseira direita [N·s/m]
        self.Cde = Cde   # Dianteira esquerda [N·s/m]
        self.Cdd = Cdd   # Dianteira direita [N·s/m]
        
        # Amortecedores estruturais do chassi
        self.Cphi = Cphi       # Torcional [N·s/m]
        self.Ctheta = Ctheta   # Flexional [N·s/m]
        
        # Inércias do chassi
        self.Iflex = Iflex     # Inércia flexional [kg·m²]
        self.Itorc = Itorc     # Inércia torcional [kg·m²]

        # Unidades de comprimento
        self.W = W       # Metade da largura do carro [m]
        self.Lt = Lt     # Distância entre eixo traseiro e CG [m]
        self.Ld = Ld     # Distância entre eixo dianteiro e CG [m]
        
        # Dicionários para armazenar interpoladores
        self.interpolators = {}
        self.chassi_interpolators = {}
        
    def spring(self, mass, k, x):
        return (k * x) / mass

    def damper(self, mass, c, xdot):
        return (c * xdot) / mass

    def preparar_interpoladores_roda(self, solucoes_rodas):
        """
        Cria funções de interpolação para o deslocamento e velocidade de cada roda.
        """
        for roda in ['FL', 'FR', 'RL', 'RR']:
            sol = solucoes_rodas[roda]
            t_eval = sol.t
            
            x_values = sol.y[0]
            xdot_values = sol.y[1]
            
            self.interpolators[f'x_{roda.lower()}'] = interp1d(t_eval, x_values, kind='linear', fill_value='extrapolate')
            self.interpolators[f'xdot_{roda.lower()}'] = interp1d(t_eval, xdot_values, kind='linear', fill_value='extrapolate')

    def preparar_interpoladores_chassi(self, solucao_chassi):
        """
        Cria funções de interpolação para os movimentos do chassi após a simulação.
        """
        t_eval = solucao_chassi.t
        phi_values, phidot_values, theta_values, thetadot_values, Xc_values, Xc_dot_values = solucao_chassi.y
        
        self.chassi_interpolators['phi'] = interp1d(t_eval, phi_values, kind='linear', fill_value='extrapolate')
        self.chassi_interpolators['phidot'] = interp1d(t_eval, phidot_values, kind='linear', fill_value='extrapolate')
        self.chassi_interpolators['theta'] = interp1d(t_eval, theta_values, kind='linear', fill_value='extrapolate')
        self.chassi_interpolators['thetadot'] = interp1d(t_eval, thetadot_values, kind='linear', fill_value='extrapolate')
        self.chassi_interpolators['Xc'] = interp1d(t_eval, Xc_values, kind='linear', fill_value='extrapolate')
        self.chassi_interpolators['Xc_dot'] = interp1d(t_eval, Xc_dot_values, kind='linear', fill_value='extrapolate')

    def car_displacements(self, t):
        """
        Calcula os deslocamentos e velocidades do chassi nos pontos de suspensão para um dado tempo 't'.
        """
        phi = self.chassi_interpolators['phi'](t)
        phidot = self.chassi_interpolators['phidot'](t)
        theta = self.chassi_interpolators['theta'](t)
        thetadot = self.chassi_interpolators['thetadot'](t)
        Xc = self.chassi_interpolators['Xc'](t)
        Xc_dot = self.chassi_interpolators['Xc_dot'](t)
        
        x_cdd = Xc + self.Ld * np.sin(theta) - self.W * np.sin(phi)
        xdot_cdd = Xc_dot + self.Ld * np.cos(theta) * thetadot - self.W * np.cos(phi) * phidot
        x_cde = Xc + self.Ld * np.sin(theta) + self.W * np.sin(phi)
        xdot_cde = Xc_dot + self.Ld * np.cos(theta) * thetadot + self.W * np.cos(phi) * phidot
        x_ctd = Xc - self.Lt * np.sin(theta) - self.W * np.sin(phi)
        xdot_ctd = Xc_dot - self.Lt * np.cos(theta) * thetadot - self.W * np.cos(phi) * phidot
        x_cte = Xc - self.Lt * np.sin(theta) + self.W * np.sin(phi)
        xdot_cte = Xc_dot - self.Lt * np.cos(theta) * thetadot + self.W * np.cos(phi) * phidot

        displacements = {
            'FL': x_cde, 'FR': x_cdd, 'RL': x_cte, 'RR': x_ctd
        }
        velocities = {
            'FL': xdot_cde, 'FR': xdot_cdd, 'RL': xdot_cte, 'RR': xdot_ctd
        }
        return displacements, velocities
    
    def dynamics_reaction(self):
        from Dynamics_metamodel import control
        roll_dot, roll_disp, pitch_dot, pitch_disp = control.model_input(degrees=True)
        print(roll_dot[1], roll_disp[1], pitch_dot[1], pitch_disp[1])
        return  roll_dot[1], roll_disp[1], pitch_dot[1], pitch_disp[1]

    def ode_chassi(self, t, y):
        # Desempacotar as variáveis de estado
        phi, phidot, theta, thetadot, Xc, Xc_dot = y
        
        # Obtendo valores interpolados para o tempo atual 't'
        x_de = self.interpolators['x_fl'](t)
        xdot_de = self.interpolators['xdot_fl'](t)
        x_dd = self.interpolators['x_fr'](t)
        xdot_dd = self.interpolators['xdot_fr'](t)
        x_te = self.interpolators['x_rl'](t)
        xdot_te = self.interpolators['xdot_rl'](t)
        x_td = self.interpolators['x_rr'](t)
        xdot_td = self.interpolators['xdot_rr'](t)

        roll_dot, roll_disp, pitch_dot, pitch_disp = 0, 0, 0, 0

        #roll_dot, roll_disp, pitch_dot, pitch_disp = self.dynamics_reaction()

        # Deslocamentos e Velocidades do Chassis em cada ponto de suspensão
        x_cdd = (
            Xc
            + self.Ld * np.sin(theta)
            - self.W * np.sin(phi)
            + self.W * np.sin(roll_disp)
            + self.Ld * np.sin(pitch_disp)
        )
        xdot_cdd = (
            Xc_dot
            + self.Ld * np.cos(theta) * thetadot
            - self.W * np.cos(phi) * phidot
            + self.W * np.cos(roll_disp) * roll_dot
            + self.Ld * np.cos(pitch_disp) * pitch_dot
        )

        x_cde = (
            Xc
            + self.Ld * np.sin(theta)
            + self.W * np.sin(phi)
            - self.W * np.sin(roll_disp)
            + self.Ld * np.sin(pitch_disp)
        )
        xdot_cde = (
            Xc_dot
            + self.Ld * np.cos(theta) * thetadot
            + self.W * np.cos(phi) * phidot
            - self.W * np.cos(roll_disp) * roll_dot
            + self.Ld * np.cos(pitch_disp) * pitch_dot
        )

        x_ctd = (
            Xc
            - self.Lt * np.sin(theta)
            - self.W * np.sin(phi)
            + self.W * np.sin(roll_disp)
            - self.Lt * np.sin(pitch_disp)
        )
        xdot_ctd = (
            Xc_dot
            - self.Lt * np.cos(theta) * thetadot
            - self.W * np.cos(phi) * phidot
            + self.W * np.cos(roll_disp) * roll_dot
            - self.Lt * np.cos(pitch_disp) * pitch_dot
        )

        x_cte = (
            Xc
            - self.Lt * np.sin(theta)
            + self.W * np.sin(phi)
            - self.W * np.sin(roll_disp)
            - self.Lt * np.sin(pitch_disp)
        )
        xdot_cte = (
            Xc_dot
            - self.Lt * np.cos(theta) * thetadot
            + self.W * np.cos(phi) * phidot
            - self.W * np.cos(roll_disp) * roll_dot
            - self.Lt * np.cos(pitch_disp) * pitch_dot
        )
        
        # Forças de suspensão (mola e amortecedor)
        F_dd_spring = self.Kdd * (x_dd - x_cdd)
        F_dd_damper = self.Cdd * (xdot_dd - xdot_cdd)
        F_de_spring = self.Kde * (x_de - x_cde)
        F_de_damper = self.Cde * (xdot_de - xdot_cde)
        F_td_spring = self.Ktd * (x_td - x_ctd)
        F_td_damper = self.Ctd * (xdot_td - xdot_ctd)
        F_te_spring = self.Kte * (x_te - x_cte)
        F_te_damper = self.Cte * (xdot_te - xdot_cte)

        F_dd = F_dd_spring + F_dd_damper
        F_de = F_de_spring + F_de_damper
        F_td = F_td_spring + F_td_damper
        F_te = F_te_spring + F_te_damper

        # Equações de movimento do chassi
        # Movimento vertical (bounce)
        Xc_acc = (F_dd + F_de + F_td + F_te) / self.Mc
        
        # Movimento de rolagem (roll)
        phi_acc = (self.W * (F_de - F_dd + F_te - F_td)) / self.Itorc - (self.Kt * phi - self.Cphi * phidot) / self.Itorc
        
        # Movimento de inclinação (pitch)
        theta_acc = (self.Ld * (F_de + F_dd) - self.Lt * (F_te + F_td)) / self.Iflex - (self.Kf * theta - self.Ctheta * thetadot) / self.Iflex

        return [phidot, phi_acc, thetadot, theta_acc, Xc_dot, Xc_acc]

    def ode_wheel(self, t, y, mass, Ks, Cs, Kt, z_road_func):
        x, xdot = y
        zr = z_road_func(t)
        
        # Aceleração da roda baseada nas forças da suspensão e do pneu
        F_susp = Ks * x + Cs * xdot
        F_tire = Kt * (x + zr)
        
        xddot = (-F_susp + F_tire) / mass
        
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

    def simular_movimento_chassi(self, t_span, t_eval):
        y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [phi, phidot, theta, thetadot, Xc, Xc_dot]
        sol = solve_ivp(fun=self.ode_chassi, t_span=t_span, y0=y0, t_eval=t_eval)
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

# =============================================================================
# Função para obter vetor de tempo
# =============================================================================

def get_simulation_time(t_start, t_end, dt):
    t_span = (t_start, t_end)
    t_eval = np.arange(t_start, t_end+dt, dt)
    return t_span, t_eval

# =============================================================================
# Execução Principal
# =============================================================================
def run_simulation():
    
    t_span, t_eval = get_simulation_time(t_start=0.0, t_end=10.0, dt=0.001)

    # Parâmetros geométricos [m]
    Ld, Lt, W = 1.4, 1.1, 0.45

    # Degraus
    altura_degrau_d = 1
    altura_degrau_e = 1
    tempo_inicio_direito = 1.0
    tempo_inicio_esquerdo = 3.0

    # Chassis
    chassis = Chassis(
        Mc=280, Mst=25, Msd=20,
        Kte=5e5, Ktd=5e5, Kde=5e5, Kdd=5e5,
        Kpte=2e4, Kptd=2e4, Kpde=2e4, Kpdd=2e4,
        Kf=3.01e7, Kt=1.6e7,
        Cte=5e3, Ctd=5e3, Cde=3e3, Cdd=3e3,
        Cphi=5e2, Ctheta=5e2,
        Iflex=5e5, Itorc=5e5,
        W=W, Lt=Lt, Ld=Ld
    )

    funcoes_entrada = chassis.gerar_funcoes_entrada(tempo_inicio_direito, tempo_inicio_esquerdo,
                                                    altura_degrau_d, altura_degrau_e)

    # PASSO 1: Simulação das rodas isoladamente
    solucoes_rodas = {}
    for roda in ['FL', 'FR', 'RL', 'RR']:
        solucoes_rodas[roda] = chassis.simular_roda(roda, t_span, t_eval, funcoes_entrada[roda])
    
    # PASSO 2: Preparar os interpoladores a partir das soluções das rodas
    chassis.preparar_interpoladores_roda(solucoes_rodas)
    
    # PASSO 3: Simular o movimento acoplado do chassi
    solucao_chassi = chassis.simular_movimento_chassi(t_span, t_eval)
    
    # PASSO 4: Preparar os interpoladores do chassi
    chassis.preparar_interpoladores_chassi(solucao_chassi)

    # Plotagem dos resultados das rodas
    chassis.plotar_resultados(solucoes_rodas, titulo='Simulação de Resposta de Rodas Isoladas (Degrau)')
    
    # Plotagem dos ângulos e do deslocamento vertical do chassi
    plt.figure(figsize=(10, 8))
    plt.plot(solucao_chassi.t, np.rad2deg(chassis.chassi_interpolators['phi'](solucao_chassi.t)), label='Ângulo de Torção (phi)')
    plt.plot(solucao_chassi.t, np.rad2deg(chassis.chassi_interpolators['theta'](solucao_chassi.t)), label='Ângulo de Inclinação (theta)')
    plt.plot(solucao_chassi.t, chassis.chassi_interpolators['Xc'](solucao_chassi.t), label='Deslocamento Vertical do Chassi (Xc)')
    plt.title('Movimento Acoplado do Chassi')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo (graus) / Deslocamento (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================================================
# Rodar a simulação
# =============================================================================
if __name__ == "__main__":
    run_simulation()

# chassis = Chassis(
#         Mc=280, Mst=25, Msd=20,
#         Kte=5e5, Ktd=5e5, Kde=5e5, Kdd=5e5,
#         Kpte=2e4, Kptd=2e4, Kpde=2e4, Kpdd=2e4,
#         Kf=3.01e7, Kt=1.6e7,
#         Cte=5e3, Ctd=5e3, Cde=3e3, Cdd=3e3,
#         Cphi=5e2, Ctheta=5e2,
#         Iflex=5e5, Itorc=5e5,
#         W=1, Lt=0.5, Ld=0.8
#     )
# chassis.dynamics_reaction()

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

    # Parâmetros geométricos [m] - ESSENCIAIS para o cálculo de momento
    lf, lr, w = 0.8, 1, 0.9

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
    
    chassis.plotar_resultados(solucoes, titulo='Simulação de Resposta de Rodas Isoladas (Degrau)')
    chassis.demonstrar_integracao(solucoes['FL'])

    # ==================================================================================
    # PASSO 3: Cálculo dos ângulos de torção e flexão (MESMO CÁLCULO ORIGINAL)
    # ==================================================================================
    print("\nAnalisando os movimentos de torção e flexão do chassi...")

    x_fl, xdot_fl = solucoes['FL'].y
    x_fr, xdot_fr = solucoes['FR'].y
    x_rl, xdot_rl = solucoes['RL'].y
    x_rr, xdot_rr = solucoes['RR'].y
    
    F_FL = chassis.Kde * x_fl + chassis.Cde * xdot_fl
    F_FR = chassis.Kdd * x_fr + chassis.Cdd * xdot_fr
    F_RL = chassis.Kte * x_rl + chassis.Cte * xdot_rl
    F_RR = chassis.Ktd * x_rr + chassis.Ctd * xdot_rr

    # Momento de Torção - CÁLCULO ORIGINAL
    M_roll = (((F_FL + F_RL) - (F_FR + F_RR))) * (w / 2)
    a_torc = M_roll / chassis.Itorc
    phi_dot, ang_torc = chassis.integrate_motion(t_eval, a_torc)

    # Momento de Flexão - CÁLCULO ORIGINAL
    M_pitch = (F_RL + F_RR) * lr - (F_FL + F_FR) * lf
    a_flex = M_pitch / chassis.Iflex
    theta_dot, ang_flex = chassis.integrate_motion(t_eval, a_flex)
    
    # ==================================================================================
    # NOVA PARTE: Aplicação dos ângulos conforme descrito na imagem
    # ==================================================================================
    print("\nAplicando efeitos dos ângulos nas rodas...")
    
    # 1. Cálculo dos deslocamentos adicionais devido aos ângulos (conforme imagem)
    desloc_torc_fl = (w/2) * np.sin(ang_torc)
    desloc_torc_fr = (w/2) * np.sin(ang_torc)
    desloc_torc_rl = (w/2) * np.sin(ang_torc)
    desloc_torc_rr = (w/2) * np.sin(ang_torc)
    
    desloc_flex_fl = lf * np.sin(ang_flex)
    desloc_flex_fr = lf * np.sin(ang_flex)
    desloc_flex_rl = lr * np.sin(ang_flex)
    desloc_flex_rr = lr * np.sin(ang_flex)
    
    # 2. Cálculo dos deslocamentos totais (original + ângulos) CREIO QUE OS CÁLCULOS ESTÃO COERENTES POREM DEVO TROCAR EM 3. OS DESLOC POR ESSE X TOTAL ou não kkkk
    x_total_fl = x_fl + desloc_torc_fl + desloc_flex_fl
    x_total_fr = x_fr + desloc_torc_fr + desloc_flex_fr
    x_total_rl = x_rl + desloc_torc_rl + desloc_flex_rl
    x_total_rr = x_rr + desloc_torc_rr + desloc_flex_rr
    
    # 3. Cálculo das forças finais (conforme fluxo da imagem) 
    F_final_fl = F_FL + chassis.Kde * desloc_torc_fl + chassis.Kde * desloc_flex_fl
    F_final_fr = F_FR + chassis.Kdd * desloc_torc_fr + chassis.Kdd * desloc_flex_fr
    F_final_rl = F_RL + chassis.Kte * desloc_torc_rl + chassis.Kte * desloc_flex_rl
    F_final_rr = F_RR + chassis.Ktd * desloc_torc_rr + chassis.Ktd * desloc_flex_rr
    
    # Momento de Torção - CÁLCULO CORRIGIDO
    M_roll = (((F_final_fl + F_final_rl) - (F_final_fr + F_final_rr))) * (w / 2)
    a_torc = M_roll / chassis.Itorc
    phi_dot, ang_torc = chassis.integrate_motion(t_eval, a_torc)

    # Momento de Flexão - CÁLCULO CORRIGIDO
    M_pitch = (F_final_rl + F_final_rr) * lr - (F_final_fl + F_final_fr) * lf
    a_flex = M_pitch / chassis.Iflex
    theta_dot, ang_flex = chassis.integrate_motion(t_eval, a_flex)
    
    # ==================================================================================
    # Plotagem dos resultados (original + novo)
    # ==================================================================================
    
    # Plotagem dos ângulos originais
    plt.figure(figsize=(12, 14))
    
    plt.subplot(4, 1, 1)
    plt.plot(t_eval, np.rad2deg(ang_torc), 'r-', label=r'Torção - $\phi$')
    plt.plot(t_eval, np.rad2deg(ang_flex), 'g-', label=r'Flexão - $\theta$')
    plt.title('Ângulos do Chassi')
    plt.ylabel('Graus')
    plt.legend()
    plt.grid(True)
    
    # Plotagem dos deslocamentos adicionais
    plt.subplot(4, 1, 2)
    plt.plot(t_eval, desloc_torc_fl, 'b-', label='Desloc Torção FL')
    plt.plot(t_eval, desloc_flex_fl, 'c-', label='Desloc Flexão FL')
    plt.title('Deslocamentos Adicionais (FL)')
    plt.ylabel('Metros')
    plt.legend()
    plt.grid(True)
    
    # Plotagem das forças finais
    plt.subplot(4, 1, 3)
    plt.plot(t_eval, F_final_fl, 'm-', label='Força Final FL')
    plt.plot(t_eval, F_final_fr, 'y-', label='Força Final FR')
    plt.title('Forças Finais nas Rodas')
    plt.ylabel('Newtons')
    plt.legend()
    plt.grid(True)
    
    # Plotagem das acelerações finais
    plt.subplot(4, 1, 4)
    plt.plot(t_eval, a_torc, 'k-', label='Aceleração Final FL')
    plt.plot(t_eval, a_flex, 'r--', label='Aceleração Final FR')
    plt.title('Acelerações Finais')
    plt.xlabel('Tempo (s)')
    plt.ylabel('m/s²')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# Rodar a simulação
# =============================================================================
if __name__ == "__main__":
    run_simulation()

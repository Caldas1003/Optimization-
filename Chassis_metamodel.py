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
        
        
    def spring(self, k, x):
        return (k * x)

    def damper(self, c, xdot):
        return (c * xdot)

    def ode_total(self, t, y, z_road_funcs):
        # -------------------------------
        # Desempacotar estados
        # -------------------------------
        x_de, xdot_de = y[0], y[1]   # Dianteiro Esquerdo
        x_dd, xdot_dd = y[2], y[3]   # Dianteiro Direito
        x_te, xdot_te = y[4], y[5]   # Traseiro Esquerdo
        x_td, xdot_td = y[6], y[7]   # Traseiro Direito

        phi, phidot     = y[8], y[9]
        theta, thetadot = y[10], y[11]
        Xc, Xc_dot      = y[12], y[13]

        # -------------------------------
        # Inputs de estrada
        # -------------------------------
        zr_de = z_road_funcs['DE'](t)
        zr_dd = z_road_funcs['DD'](t)
        zr_te = z_road_funcs['TE'](t)
        zr_td = z_road_funcs['TD'](t)

        # -------------------------------
        # Cinemática do chassi
        # -------------------------------
        
        x_cdd = Xc -(+ self.Ld*np.sin(theta) - self.W*np.sin(phi))   # dianteiro direito
        x_cde = Xc -(+ self.Ld*np.sin(theta) + self.W*np.sin(phi))   # dianteiro esquerdo
        x_ctd = Xc -(- self.Lt*np.sin(theta) - self.W*np.sin(phi))   # traseiro direito
        x_cte = Xc -(- self.Lt*np.sin(theta) + self.W*np.sin(phi))   # traseiro esquerdo

        xdot_cdd = Xc_dot -(+ self.Ld*np.cos(theta)*thetadot - self.W*np.cos(phi)*phidot)
        xdot_cde = Xc_dot -(+ self.Ld*np.cos(theta)*thetadot + self.W*np.cos(phi)*phidot)
        xdot_ctd = Xc_dot -(- self.Lt*np.cos(theta)*thetadot - self.W*np.cos(phi)*phidot)
        xdot_cte = Xc_dot -(- self.Lt*np.cos(theta)*thetadot + self.W*np.cos(phi)*phidot)

        print(self.Ld*np.sin(theta) - self.W*np.sin(phi))
        

        # -------------------------------
        # Forças de suspensão
        # -------------------------------
        F_de = self.Kde*(x_cde) + self.Cde*(xdot_cde)
        F_dd = self.Kdd*(x_cdd) + self.Cdd*(xdot_cdd)
        F_te = self.Kte*(x_cte) + self.Cte*(xdot_cte)
        F_td = self.Ktd*(x_ctd) + self.Ctd*(xdot_ctd)

        # -------------------------------
        # Dinâmica das rodas
        # -------------------------------
        xddot_de = (-self.Kde*(x_de - x_cde) - self.Cde*(xdot_de - xdot_cde) - self.Kpde*(x_de - zr_de)) / self.Msd_unsprung
        xddot_dd = (-self.Kdd*(x_dd - x_cdd) - self.Cdd*(xdot_dd - xdot_cdd) - self.Kpdd*(x_dd - zr_dd)) / self.Msd_unsprung
        xddot_te = (-self.Kte*(x_te - x_cte) - self.Cte*(xdot_te - xdot_cte) - self.Kpte*(x_te - zr_te)) / self.Mst_unsprung
        xddot_td = (-self.Ktd*(x_td - x_ctd) - self.Ctd*(xdot_td - xdot_ctd) - self.Kptd*(x_td - zr_td)) / self.Mst_unsprung

        # -------------------------------
        # Dinâmica do chassi
        # -------------------------------
        Xc_acc = -(F_de + F_dd)/ self.Msd_unsprung - (F_te + F_td)/self.Mst_unsprung
        
        phi_acc = ((self.W / self.Itorc) * (
            + (-self.Kde * x_de) + ( self.Kdd * x_dd) + (-self.Kte * x_te) + ( self.Ktd * x_td)                   # Termos das rigidezes multiplicando deslocamentos das rodas
            + ( self.Kde * x_cde) + (-self.Kdd * x_cdd) + ( self.Kte * x_cte) + (-self.Ktd * x_ctd)               # Termos das rigidezes multiplicando deslocamentos do chassi nos pontos
            + (-self.Cde * xdot_de) + ( self.Cdd * xdot_dd) + (-self.Cte * xdot_te) + ( self.Ctd * xdot_td)       # Termos dos amortecedores multiplicando velocidades das rodas
            + ( self.Cde * xdot_cde) + (-self.Cdd * xdot_cdd) + ( self.Cte * xdot_cte) + (-self.Ctd * xdot_ctd))  # Termos dos amortecedores multiplicando velocidades do chassi nos pontos
            - (self.Kt * phi + self.Cphi * phidot) / self.Itorc                                                   # Termo estruturais (rigidez torcional e amortecimento torcional)
            )

        theta_acc = (1 / self.Iflex) * (
            # Contribuição das rigidezes (Ld * (de + dd) - Lt * (te + td))
            + self.Ld * ( self.Kde * x_de + self.Kdd * x_dd - self.Kde * x_cde - self.Kdd * x_cdd )
            - self.Lt * ( self.Kte * x_te + self.Ktd * x_td - self.Kte * x_cte - self.Ktd * x_ctd )
            # Contribuição dos amortecedores (Ld * (...) - Lt * (...))
            + self.Ld * ( self.Cde * xdot_de + self.Cdd * xdot_dd - self.Cde * xdot_cde - self.Cdd * xdot_cdd )
            - self.Lt * ( self.Cte * xdot_te + self.Ctd * xdot_td - self.Cte * xdot_cte - self.Ctd * xdot_ctd )
            # Termos estruturais (rigidez flexional e amortecimento flexional)
            - (self.Kf * theta + self.Ctheta * thetadot)
        )

        # -------------------------------
        # Retornar Funções
        # -------------------------------
        return [
            xdot_de, xddot_de,
            xdot_dd, xddot_dd,
            xdot_te, xddot_te,
            xdot_td, xddot_td,
            phidot, phi_acc,
            thetadot, theta_acc,
            Xc_dot, Xc_acc
        ]
    
    def dynamics_reaction(self):
        from Dynamics_metamodel import control
        roll_dot, roll_disp, pitch_dot, pitch_disp = control.model_input(degrees=True)
        print(roll_dot[1], roll_disp[1], pitch_dot[1], pitch_disp[1])
        return  roll_dot[1], roll_disp[1], pitch_dot[1], pitch_disp[1]

    def gerar_funcoes_entrada(self, tempo_direito, tempo_esquerdo, step_value_d, step_value_e):
        def road_input_step(t, t_start, height):
            return height if t >= t_start else 0.0
        func_direito  = functools.partial(road_input_step, t_start=tempo_direito, height=step_value_d)
        func_esquerdo = functools.partial(road_input_step, t_start=tempo_esquerdo, height=step_value_e)
        return {'DE': func_esquerdo, 'DD': func_direito,
                'TE': func_esquerdo, 'TD': func_direito}
                

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
        Cphi=5e4, Ctheta=5e4,
        Iflex=5e5, Itorc=5e5,
        W=W, Lt=Lt, Ld=Ld
    )

    funcoes_entrada = chassis.gerar_funcoes_entrada(tempo_inicio_direito, tempo_inicio_esquerdo,
                                                    altura_degrau_d, altura_degrau_e)
    
    y0 = np.zeros(14)  # estados iniciais (rodas e chassi em repouso)
    sol = solve_ivp(fun=lambda t,y: chassis.ode_total(t,y,funcoes_entrada),
                t_span=t_span, y0=y0, t_eval=t_eval, method='LSODA')
    
    # Extraindo soluções individuais de cada variável do sol
    t = sol.t  # vetor de tempo

    # Exemplo de mapeamento (adapte conforme sua ordem de estados):
    # 0 → roda dianteira esquerda
    # 1 → roda dianteira direita
    # 2 → roda traseira esquerda
    # 3 → roda traseira direita
    # 4 → torção do chassi
    # 5 → flexão do chassi
    # 6 → deslocamento global do carro

    solucao_roda_de = sol.y[0]   # dianteira esquerda
    solucao_roda_dd = sol.y[2]   # dianteira direita
    solucao_roda_te = sol.y[4]   # traseira esquerda
    solucao_roda_td = sol.y[6]   # traseira direita
    solucao_torcao  = sol.y[8]   # torção do chassi (phi)
    solucao_flexao  = sol.y[10]  # flexão do chassi (theta)
    solucao_carro   = sol.y[12]  # deslocamento global do carro


    plt.plot(t, solucao_roda_de, label="Roda Dianteira Esquerda")
    plt.plot(t, solucao_roda_dd, label="Roda Dianteira Direita")
    plt.plot(t, solucao_roda_te, label="Roda Traseira Esquerda")
    plt.plot(t, solucao_roda_td, label="Roda Traseira Direita")
    plt.legend()
    plt.xlabel("Tempo [s]")
    plt.ylabel("Deslocamento [m]")
    plt.show()

    plt.plot(t, solucao_torcao, label="Torção do Chassi")
    plt.plot(t, solucao_flexao, label="Flexão do Chassi")
    plt.plot(t, solucao_carro, label="Deslocamento Global do Carro")
    plt.legend()
    plt.xlabel("Tempo [s]")
    plt.ylabel("Deslocamento [m]")
    plt.show()

# =============================================================================
# Rodar a simulação
# =============================================================================
if __name__ == "__main__":
    run_simulation()

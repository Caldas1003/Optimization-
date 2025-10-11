import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import functools

# =============================================================================
# Modelo do Chassis com todas as funções internas organizadas
# =============================================================================
class TireModel:
    def __init__(self, tire_Sa=0, front_tire_Ls=None, rear_tire_Ls=None, tire_friction_coef=None, tire_Ca=0, params=None):
        self.front_tire_Sa = tire_Sa                  # Ângulo de deslizamento lateral do pneu [rad]
        self.rear_tire_Sa = tire_Sa / 10
        self.front_tire_Ls = front_tire_Ls            # Escorregamento longitudinal frontal [adimensional]
        self.rear_tire_Ls = rear_tire_Ls              # Escorregamento longitudinal traseiro [adimensional]
        self.tire_friction_coef = tire_friction_coef  # Coeficiente de atrito entre pneu e pista
        self.tire_Ca = tire_Ca                        # Ângulo de camber do pneu
        self.params = params                          # Parâmetros de Pacejka (E, Cy, Cx, c1, c2)

    def pacejka_params(self, Fz):
        # Desembalando os parâmetros de Pacejka
        E, Cy, Cx, c1, c2 = self.params
        # Calculando parâmetros intermediários
        Cs = c1 * np.sin(2 * np.arctan(Fz / c2))
        D = self.tire_friction_coef * Fz
        Bx = Cs / (Cx * D)
        By = Cs / (Cy * D)
        return E, Cy, Cx, Bx, By, D

    def lateral_force(self, Fz, slip_angle, camber=0):
        # Desembalando os parâmetros de Pacejka
        E, Cy, Cx, Bx, By, D = TireModel.pacejka_params(self, Fz)
        # Calculando a força lateral do pneu
        tire_lateral_force = D * np.sin(Cy * np.arctan(By * slip_angle - E * (By * slip_angle - np.arctan(By * slip_angle))))
        # Calculando a força de camber
        camber_thrust = D / 2 * np.sin(Cy * np.arctan(By * camber))
        return tire_lateral_force + camber_thrust

    def longitudinal_force(self, Fz, slip_ratio):
        # Desembalando os parâmetros de Pacejka
        E, Cy, Cx, Bx, By, D = TireModel.pacejka_params(self, Fz)
        # Calculando a força longitudinal do pneu
        tire_longitudinal_force = D * np.sin(Cx * np.arctan(9 * Bx * slip_ratio - E * (Bx * slip_ratio - np.arctan(Bx * slip_ratio))))
        return tire_longitudinal_force

class VehicleDynamics:
    def __init__(self, mass, Iz, Ix, Iy, Ld, Lt, W, hcg, tire_params=0, tire_friction_coef=1.0, tire_Ca=0.0):
        self.m = mass
        self.Iz = Iz   # momento de inércia em yaw
        self.Ix = Ix   # momento de inércia em roll
        self.Iy = Iy   # momento de inércia em pitch
        self.Ld = Ld   # distância CG → eixo dianteiro
        self.Lt = Lt   # distância CG → eixo traseiro
        self.W = W     # bitola (track width)
        self.hcg = hcg # altura do CG

        # Modelo de pneus
        self.tires = TireModel(
            tire_friction_coef=tire_friction_coef,
            tire_Ca=tire_Ca,
            params=tire_params
        )

    def yaw_acc(self, Fx_de=0, Fx_dd=0, Fx_te=0, Fx_td=0, Fy_de=0, Fy_dd=0, Fy_te=0, Fy_td=0):
        # Momentos
        lateral_moment = (Fy_de + Fy_dd) * self.Ld - (Fy_te + Fy_td) * self.Lt
        longitudinal_moment = (Fx_de - Fx_dd + Fx_te - Fx_td) * (self.W / 2)
        return (lateral_moment + longitudinal_moment) / self.Iz

    def roll_acc(self, Fy_de=0, Fy_dd=0, Fy_te=0, Fy_td=0, chassis_F_de=0, chassis_F_dd=0, chassis_F_te=0, chassis_F_td=0):
        roll_moment = (Fy_de + Fy_dd + Fy_te + Fy_td) * self.hcg - (chassis_F_de + chassis_F_dd + chassis_F_te + chassis_F_td) * self.W
        return roll_moment / self.Ix, roll_moment

    def pitch_acc(self, Fx_de=0, Fx_dd=0, Fx_te=0, Fx_td=0, chassis_F_de=0, chassis_F_dd=0, chassis_F_te=0, chassis_F_td=0):
        pitch_moment = (Fx_de + Fx_dd + Fx_te + Fx_td) * self.hcg - (((chassis_F_de + chassis_F_dd) * self.Ld) + ((chassis_F_te + chassis_F_td) * self.Lt)) 
        return pitch_moment / self.Iy, pitch_moment
    
    def load_transfer(self, Fz_static_rear, Fz_static_front, 
                    Fx_de=0, Fx_dd=0, Fx_te=0, Fx_td=0,
                    Fy_de=0, Fy_dd=0, Fy_te=0, Fy_td=0):
        """
        Atualiza as cargas verticais (Fz) em cada roda considerando
        transferência dinâmica de carga por rolagem e arfagem.
        """

        # Cargas adicionais devido à dinâmica
        pitch_load = (Fx_de + Fx_dd + Fx_te + Fx_td) * self.hcg
        roll_load = (Fy_de + Fy_dd + Fy_te + Fy_td) * self.hcg

        # Atualizando as forças normais em tempo real
        Tire_Fz_dd = Fz_static_front/2 + roll_load - pitch_load
        Tire_Fz_de = Fz_static_front/2 - roll_load - pitch_load
        Tire_Fz_td = Fz_static_rear/2 + roll_load + pitch_load
        Tire_Fz_te = Fz_static_rear/2 - roll_load + pitch_load

        return Tire_Fz_dd, Tire_Fz_de, Tire_Fz_td, Tire_Fz_te
    
class FullModel:
    def __init__(self, 
                Mc=0, Mst=None, Msd=None, Distribution=0,   # Massas
                Kte=None, Ktd=None, Kde=None, Kdd=None,           # Rigidez suspensão
                Kpte=None, Kptd=None, Kpde=None, Kpdd=None,       # Rigidez pneus
                Kf=None, Kt=None,                                 # Rigidez do chassi
                Cte=None, Ctd=None, Cde=None, Cdd=None,           # Amortecedores suspensão
                Cphi=None, Ctheta=None,                           # Amortecedores do chassi
                Iflex=None, Itorc=None,                           # Inércias do chassi
                W=None, Lt=None, Ld=None, hcg=None,               # Unidade de comprimento
                tire_coef=None,params=None,                       # Parâmetros de pneu
                slip=None, rear_ratio=None, front_ratio=None,     # Entradas de controle do carro
                Ix=None, Iz=None, Iy=None
                ):                                      
        
        # Massas
        self.Mc = Mc                                              # Massa suspensa TOTAL [kg]
        self.Mst_unsprung = Mst                                   # Massa NÃO suspensa traseira [kg]
        self.Msd_unsprung = Msd                                   # Massa NÃO suspensa dianteira [kg]
        self.Distribution = Distribution                          # Distribuição de peso na traseira (adimensional)
        self.Fz_static_rear = 9.81 * Mc * Distribution/2          # Carga no eixo traseiro (N)
        self.Fz_static_front = 9.81 * Mc * (1 - Distribution)/2   # Carga no eixo dianteiro (N)
        
        
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

        # Entradas de controle do carro
        self.slip_angle = slip                  # Ângulo de esterçamento [Graus]
        self.rear_slip_angle = slip/10          # Ângulo de esterçamento traseiro [Graus]
        self.rear_slip_ratio = rear_ratio       # Taxa de escorregamento traseira [Adm]
        self.front_slip_ratio = front_ratio     # Taxa de escorregamento dianteira [Adm]


        self.tire = TireModel(
            tire_friction_coef = tire_coef,
            params=params)
        
        self.dynamics = VehicleDynamics(mass=Mc, Iz=Iz, Ix=Ix, Iy=Iy, Ld=Ld, Lt=Lt, W=W, hcg=hcg)
        
        
    def spring(self, k, x):
        return (k * x)

    def damper(self, c, xdot):
        return (c * xdot)

    def ode_total(self, t, y, z_road_funcs): 

        " ------------------------------- Metamodelo de Chassis------------------------------- "
        # -------------------------------
        # Desempacotar estados de Chassi
        # -------------------------------
        x_de, xdot_de = y[0], y[1]   # Dianteiro Esquerdo
        x_dd, xdot_dd = y[2], y[3]   # Dianteiro Direito
        x_te, xdot_te = y[4], y[5]   # Traseiro Esquerdo
        x_td, xdot_td = y[6], y[7]   # Traseiro Direito

        phi, phidot     = y[8], y[9]
        theta, thetadot = y[10], y[11]
        Xc, Xc_dot      = y[12], y[13]

        # ----------------------------
        # Desempacotar estados de Dinâmica
        # ----------------------------
        yaw, yawdot = y[14], y[15]       # Yaw
        pitch, pitchdot = y[16], y[17]   # Pitch(Guinada)
        roll, rolldot = y[18], y[19]     # Roll(Rolagem)
        Fz_de, Fz_dd, Fz_te, Fz_td = y[20], y[21], y[22], y[23]
        x_car, y_car = y[24], y[25]
        vx_car, vy_car = y[25], y[27]

        # -------------------------------
        # Inputs de estrada
        # -------------------------------
        zr_de = z_road_funcs['DE'](t)
        zr_dd = z_road_funcs['DD'](t)
        zr_te = z_road_funcs['TE'](t)
        zr_td = z_road_funcs['TD'](t)

        xreac_dd = -(self.W * np.sin(roll)) + (self.Ld * np.sin(pitch))
        xreac_de = +(self.W * np.sin(roll)) + (self.Ld * np.sin(pitch))
        xreac_td = -(self.W * np.sin(roll)) - (self.Lt * np.sin(pitch))
        xreac_te = +(self.W * np.sin(roll)) - (self.Lt * np.sin(pitch))

        xdotreac_dd = -((self.W * np.sin(roll))* rolldot) + ((self.Ld * np.sin(pitch))* pitchdot)
        xdotreac_de = +((self.W * np.sin(roll))* rolldot) + ((self.Ld * np.sin(pitch))* pitchdot)
        xdotreac_td = -((self.W * np.sin(roll))* rolldot) - ((self.Lt * np.sin(pitch))* pitchdot)
        xdotreac_te = +((self.W * np.sin(roll))* rolldot) - ((self.Lt * np.sin(pitch))* pitchdot)

        # -------------------------------
        # Cinemática do chassi
        # -------------------------------
        
        x_cdd = Xc -(- self.Ld*np.sin(theta) - self.W*np.sin(phi)) + xreac_dd   # dianteiro direito
        x_cde = Xc -(- self.Ld*np.sin(theta) + self.W*np.sin(phi)) + xreac_de   # dianteiro esquerdo
        x_ctd = Xc -(+ self.Lt*np.sin(theta) - self.W*np.sin(phi)) + xreac_td   # traseiro direito
        x_cte = Xc -(+ self.Lt*np.sin(theta) + self.W*np.sin(phi)) + xreac_te   # traseiro esquerdo

        xdot_cdd = Xc_dot -(- self.Ld*np.cos(theta)*thetadot - self.W*np.cos(phi)*phidot - xdotreac_dd)
        xdot_cde = Xc_dot -(- self.Ld*np.cos(theta)*thetadot + self.W*np.cos(phi)*phidot - xdotreac_de)
        xdot_ctd = Xc_dot -(+ self.Lt*np.cos(theta)*thetadot - self.W*np.cos(phi)*phidot - xdotreac_td)
        xdot_cte = Xc_dot -(+ self.Lt*np.cos(theta)*thetadot + self.W*np.cos(phi)*phidot - xdotreac_te)

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

        " ------------------------------- Metamodelo de Dinâmica------------------------------- "
        # ----------------------------
        # Forças dos pneus
        # ----------------------------

        Fx_de = self.tire.longitudinal_force(Fz_de, self.front_slip_ratio)
        Fx_dd = self.tire.longitudinal_force(Fz_dd, self.front_slip_ratio)
        Fx_te = self.tire.longitudinal_force(Fz_te, self.rear_slip_ratio)
        Fx_td = self.tire.longitudinal_force(Fz_td, self.rear_slip_ratio)

        Fy_de = self.tire.lateral_force(Fz_de, self.slip_angle)
        Fy_dd = self.tire.lateral_force(Fz_dd, self.slip_angle)
        Fy_te = self.tire.lateral_force(Fz_te, self.rear_slip_angle)
        Fy_td = self.tire.lateral_force(Fz_td, self.rear_slip_angle)

        Fz_dd, Fz_de, Fz_td, Fz_te = self.dynamics.load_transfer(
                                    Fz_static_front=self.Fz_static_front, Fz_static_rear=self.Fz_static_rear,
                                    Fx_de=Fx_de, Fx_dd=Fx_dd, Fx_te=Fx_te, Fx_td=Fx_td,
                                    Fy_de=Fy_de, Fy_dd=Fy_dd, Fy_te=Fy_te, Fy_td=Fy_td)
        # ----------------------------
        # Acelerações Angulares
        # ----------------------------

        yawddot = self.dynamics.yaw_acc(Fx_de=Fx_de, Fx_dd=Fx_dd, Fx_te=Fx_te, Fx_td=Fx_td, Fy_de=Fy_de, Fy_dd=Fy_dd, Fy_te=Fy_te, Fy_td=Fy_td)
        rollddot, _ = self.dynamics.roll_acc(Fy_de=Fy_de, Fy_dd=Fy_dd, Fy_te=Fy_te, Fy_td=Fy_td, chassis_F_de=F_de, chassis_F_dd=F_dd, chassis_F_te=F_te, chassis_F_td=F_td)
        pitchddot, _ = self.dynamics.pitch_acc(Fx_de=Fx_de, Fx_dd=Fx_dd, Fx_te=Fx_te, Fx_td=Fx_td, chassis_F_de=F_de, chassis_F_dd=F_dd, chassis_F_te=F_te, chassis_F_td=F_td)

        x_acc_car = ((Fx_de + Fx_dd + Fx_te + Fx_td) * np.cos(yaw) + (Fy_de + Fy_dd + Fy_te + Fy_td) * np.sin(yaw))/ self.Mc
        y_acc_car = ((Fy_de + Fy_dd + Fy_te + Fy_td) * np.cos(yaw) + (Fx_de + Fx_dd + Fx_te + Fx_td) * np.sin(yaw))/ self.Mc

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
            Xc_dot, Xc_acc,
            yawdot, yawddot,
            pitchdot, pitchddot,
            rolldot, rollddot,
            Fz_de, Fz_dd, Fz_te, Fz_td,
            vx_car, vy_car,
            x_acc_car, y_acc_car
        ]

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
    
    t_span, t_eval = get_simulation_time(t_start=0.0, t_end=10.0, dt=0.00001)

    # Parâmetros geométricos [m]
    Ld, Lt, W = 1.4, 1.1, 0.5

    # Degraus
    altura_degrau_d = 1
    altura_degrau_e = 1
    tempo_inicio_direito = 1.0
    tempo_inicio_esquerdo = 3.0

    # Chassis (exemplo de parâmetros; ajuste conforme quiser)
    chassis = FullModel(
        Mc=280, Mst=25, Msd=20, Distribution=0.5,
        Kte=5e5, Ktd=5e5, Kde=5e5, Kdd=5e5,
        Kpte=2e4, Kptd=2e4, Kpde=2e4, Kpdd=2e4,
        Kf=3.01e7, Kt=1.6e7,
        Cte=5e3, Ctd=5e3, Cde=3e4, Cdd=3e3,
        Cphi=2e5, Ctheta=2e5,
        Iflex=5e5, Itorc=5e5,
        W=W, Lt=Lt, Ld=Ld, hcg=0.3,
        tire_coef=1.45, params=(0.3336564873588197, 1.627, 1.0, 931.405, 366.493),
        slip=9, rear_ratio=0.25, front_ratio=0.10,
        Ix=1e5, Iz=1e3, Iy=1e5
    )

    funcoes_entrada = chassis.gerar_funcoes_entrada(tempo_inicio_direito, tempo_inicio_esquerdo,
                                                    altura_degrau_d, altura_degrau_e)
    
    Fz_de0 = 600
    Fz_dd0 = 600
    Fz_te0 = 800
    Fz_td0 = 800

    y0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Fz_de0, Fz_dd0, Fz_te0, Fz_td0,0,0,0,0]
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

    solucao_roda_de = sol.y[0]    # dianteira esquerda
    solucao_roda_dd = sol.y[2]    # dianteira direita
    solucao_roda_te = sol.y[4]    # traseira esquerda
    solucao_roda_td = sol.y[6]    # traseira direita
    solucao_torcao  = sol.y[8]    # torção do chassi (phi)
    solucao_flexao  = sol.y[10]   # flexão do chassi (theta)
    solucao_carro   = sol.y[12]   # deslocamento global do carro

    # Novos parâmetros dinâmicos
    solucao_yaw     = sol.y[14]   # guinada
    solucao_roll    = sol.y[16]   # rolagem
    solucao_pitch   = sol.y[18]   # arfagem

    # Deslocamentos e velocidades finais do carro
    solucao_x       = sol.y[24]
    solucao_y       = sol.y[25]
    solucao_vx      = sol.y[26]
    solucao_vy      = sol.y[27]

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

    # Yaw
    plt.figure()
    plt.plot(t, solucao_yaw, label="Yaw (guinada)")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Ângulo [rad]")
    plt.title("Resposta em Yaw")
    plt.legend()
    plt.grid(True)

    # Roll
    plt.figure()
    plt.plot(t, solucao_roll, label="Roll (rolagem)")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Ângulo [rad]")
    plt.title("Resposta em Roll")
    plt.legend()
    plt.grid(True)

    # Pitch
    plt.figure()
    plt.plot(t, solucao_pitch, label="Pitch (arfagem)")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Ângulo [rad]")
    plt.title("Resposta em Pitch")
    plt.legend()
    plt.grid(True)

    # Deslocamento
    plt.figure()
    plt.plot(t, solucao_x, label="Deslocamento em X")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Distância [m]")
    plt.title("Deslocamento do carro")
    plt.legend()
    plt.grid(True)

    # Deslocamento
    plt.figure()
    plt.plot(t, solucao_y, label="Deslocamento em Y")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Distância [m]")
    plt.title("Deslocamento do carro")
    plt.legend()
    plt.grid(True)

    # Deslocamento
    plt.figure()
    plt.plot(t, solucao_vx, label="Velocidade em X")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Velocidade [m/s]")
    plt.title("Velocidade do carro")
    plt.legend()
    plt.grid(True)

    # Deslocamento
    plt.figure()
    plt.plot(t, solucao_vy, label="Velocidade em Y")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Velocidade [m/s]")
    plt.title("Velocidade do carro")
    plt.legend()
    plt.grid(True)

# =============================================================================
# Rodar a simulação
# =============================================================================
if __name__ == "__main__":
    run_simulation()

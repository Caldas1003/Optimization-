import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import functools

# =============================================================================
# Modelo do Chassis
# =============================================================================
class TireModel:
    def __init__(self, tire_Sa=0, front_tire_Ls=None, rear_tire_Ls=None, tire_friction_coef=None, tire_Ca=0, params=None):
        self.front_tire_Sa = tire_Sa                  
        self.rear_tire_Sa = tire_Sa / 10
        self.front_tire_Ls = front_tire_Ls            
        self.rear_tire_Ls = rear_tire_Ls              
        self.tire_friction_coef = tire_friction_coef  
        self.tire_Ca = tire_Ca                        
        self.params = params                          

    def pacejka_params(self, Fz):
        E, Cy, Cx, c1, c2 = self.params
        Cs = c1 * np.sin(2 * np.arctan(Fz / c2))
        D = self.tire_friction_coef * Fz
        Bx = Cs / (Cx * D)
        By = Cs / (Cy * D)
        return E, Cy, Cx, Bx, By, D

    def lateral_force(self, Fz, slip_angle, camber=0):
        E, Cy, Cx, Bx, By, D = TireModel.pacejka_params(self, Fz)
        tire_lateral_force = D * np.sin(Cy * np.arctan(By * slip_angle - E * (By * slip_angle - np.arctan(By * slip_angle))))
        camber_thrust = D / 2 * np.sin(Cy * np.arctan(By * camber))
        return tire_lateral_force + camber_thrust

    def longitudinal_force(self, Fz, slip_ratio):
        E, Cy, Cx, Bx, By, D = TireModel.pacejka_params(self, Fz)
        tire_longitudinal_force = D * np.sin(Cx * np.arctan(9 * Bx * slip_ratio - E * (Bx * slip_ratio - np.arctan(Bx * slip_ratio))))
        return tire_longitudinal_force

class VehicleDynamics:
    def __init__(self, mass, Iz, Ix, Iy, Ld, Lt, W, hcg, tire_params=0, tire_friction_coef=1.0, tire_Ca=0.0):
        self.m = mass
        self.Iz = Iz   
        self.Ix = Ix   
        self.Iy = Iy   
        self.Ld = Ld   
        self.Lt = Lt   
        self.W = W     
        self.hcg = hcg 

        self.tires = TireModel(
            tire_friction_coef=tire_friction_coef,
            tire_Ca=tire_Ca,
            params=tire_params
        )

    def yaw_acc(self, Fx_de=0, Fx_dd=0, Fx_te=0, Fx_td=0, Fy_de=0, Fy_dd=0, Fy_te=0, Fy_td=0):
        lateral_moment = (Fy_de + Fy_dd) * self.Ld - (Fy_te + Fy_td) * self.Lt
        longitudinal_moment = (Fx_de - Fx_dd + Fx_te - Fx_td) * (self.W) 
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
        
        pitch_moment_total = (Fx_de + Fx_dd + Fx_te + Fx_td) * self.hcg
        roll_moment_total = (Fy_de + Fy_dd + Fy_te + Fy_td) * self.hcg

        wheelbase = self.Ld + self.Lt
        track_width = self.W * 2 

        pitch_load_delta = pitch_moment_total / wheelbase
        roll_load_delta = roll_moment_total / track_width

        Tire_Fz_dd = Fz_static_front/2 + roll_load_delta - pitch_load_delta
        Tire_Fz_de = Fz_static_front/2 - roll_load_delta - pitch_load_delta
        Tire_Fz_td = Fz_static_rear/2  + roll_load_delta + pitch_load_delta
        Tire_Fz_te = Fz_static_rear/2  - roll_load_delta + pitch_load_delta

        return Tire_Fz_dd, Tire_Fz_de, Tire_Fz_td, Tire_Fz_te
    
class FullModel:
    def __init__(self, 
                Mc=0, Mst=None, Msd=None, Distribution=0,   
                Kte=None, Ktd=None, Kde=None, Kdd=None,           
                Kpte=None, Kptd=None, Kpde=None, Kpdd=None,       
                Kf=None, Kt=None,                                 
                Cte=None, Ctd=None, Cde=None, Cdd=None,           
                Cphi=None, Ctheta=None,                           
                Iflex=None, Itorc=None,                           
                W=None, Lt=None, Ld=None, hcg=None,               
                tire_coef=None,params=None,                       
                Fx_total=0, Fy_total=0,                           
                Ix=None, Iz=None, Iy=None
                ):                                      
        
        # Massas e Parâmetros
        self.Mc = Mc                                              
        self.Mst_unsprung = Mst                                   
        self.Msd_unsprung = Msd                                   
        self.Distribution = Distribution                          
        self.Fz_static_rear = 9.81 * Mc * Distribution/2          
        self.Fz_static_front = 9.81 * Mc * (1 - Distribution)/2   
        
        self.Kte, self.Ktd, self.Kde, self.Kdd = Kte, Ktd, Kde, Kdd
        self.Kpte, self.Kptd, self.Kpde, self.Kpdd = Kpte, Kptd, Kpde, Kpdd
        self.Kf, self.Kt = Kf, Kt
        self.Cte, self.Ctd, self.Cde, self.Cdd = Cte, Ctd, Cde, Cdd
        self.Cphi, self.Ctheta = Cphi, Ctheta
        self.Iflex, self.Itorc = Iflex, Itorc
        self.W, self.Lt, self.Ld = W, Lt, Ld
        self.Fx_total, self.Fy_total = Fx_total, Fy_total

        self.tire = TireModel(tire_friction_coef = tire_coef, params=params)
        self.dynamics = VehicleDynamics(mass=Mc, Iz=Iz, Ix=Ix, Iy=Iy, Ld=Ld, Lt=Lt, W=W, hcg=hcg)
        
    def ode_total(self, t, y, z_road_funcs): 

        " ------------------------------- Metamodelo de Chassis ------------------------------- "
        x_de, xdot_de = y[0], y[1]   
        x_dd, xdot_dd = y[2], y[3]   
        x_te, xdot_te = y[4], y[5]   
        x_td, xdot_td = y[6], y[7]   

        phi, phidot     = y[8], y[9]
        theta, thetadot = y[10], y[11]
        Xc, Xc_dot      = y[12], y[13]

        yaw, yawdot = y[14], y[15]       
        pitch, pitchdot = y[16], y[17]   
        roll, rolldot = y[18], y[19]     
        
        Fz_de_curr, Fz_dd_curr, Fz_te_curr, Fz_td_curr = y[20], y[21], y[22], y[23] 
        
        x_car, y_car = y[24], y[25]
        vx_car, vy_car = y[26], y[27]

        zr_de = z_road_funcs['DE'](t)
        zr_dd = z_road_funcs['DD'](t)
        zr_te = z_road_funcs['TE'](t)
        zr_td = z_road_funcs['TD'](t)

        # -------------------------------
        # Cinemática do chassi (SIMPLIFICADA E CORRIGIDA)
        # -------------------------------
        # Removemos 'xreac' e calculamos a posição e velocidade dos pontos de fixação (hardpoints)
        # diretamente. Isso garante que a Velocidade seja exatamente a derivada da Posição.
        
        # Dianteira Esquerda: Sobe com Pitch (+Ld), Sobe com Roll (+W)
        x_cde = Xc + self.Ld * np.sin(theta) + self.W * np.sin(phi)
        xdot_cde = Xc_dot + self.Ld * np.cos(theta) * thetadot + self.W * np.cos(phi) * phidot

        # Dianteira Direita: Sobe com Pitch (+Ld), Desce com Roll (-W)
        x_cdd = Xc + self.Ld * np.sin(theta) - self.W * np.sin(phi)
        xdot_cdd = Xc_dot + self.Ld * np.cos(theta) * thetadot - self.W * np.cos(phi) * phidot

        # Traseira Esquerda: Desce com Pitch (-Lt), Sobe com Roll (+W)
        x_cte = Xc - self.Lt * np.sin(theta) + self.W * np.sin(phi)
        xdot_cte = Xc_dot - self.Lt * np.cos(theta) * thetadot + self.W * np.cos(phi) * phidot

        # Traseira Direita: Desce com Pitch (-Lt), Desce com Roll (-W)
        x_ctd = Xc - self.Lt * np.sin(theta) - self.W * np.sin(phi)
        xdot_ctd = Xc_dot - self.Lt * np.cos(theta) * thetadot - self.W * np.cos(phi) * phidot

        # -------------------------------
        # Forças de suspensão
        # -------------------------------
        F_de = self.Kde*(x_cde) + self.Cde*(xdot_cde)
        F_dd = self.Kdd*(x_cdd) + self.Cdd*(xdot_cdd)
        F_te = self.Kte*(x_cte) + self.Cte*(xdot_cte)
        F_td = self.Ktd*(x_ctd) + self.Ctd*(xdot_ctd)

        # Dinâmica das rodas
        xddot_de = (-self.Kde*(x_de - x_cde) - self.Cde*(xdot_de - xdot_cde) - self.Kpde*(x_de - zr_de)) / self.Msd_unsprung
        xddot_dd = (-self.Kdd*(x_dd - x_cdd) - self.Cdd*(xdot_dd - xdot_cdd) - self.Kpdd*(x_dd - zr_dd)) / self.Msd_unsprung
        xddot_te = (-self.Kte*(x_te - x_cte) - self.Cte*(xdot_te - xdot_cte) - self.Kpte*(x_te - zr_te)) / self.Mst_unsprung
        xddot_td = (-self.Ktd*(x_td - x_ctd) - self.Ctd*(xdot_td - xdot_ctd) - self.Kptd*(x_td - zr_td)) / self.Mst_unsprung

        # Dinâmica do chassi
        Xc_acc = -(F_de + F_dd)/ self.Msd_unsprung - (F_te + F_td)/self.Mst_unsprung
        
        phi_acc = ((self.W / self.Itorc) * (
            + (-self.Kde * x_de) + ( self.Kdd * x_dd) + (-self.Kte * x_te) + ( self.Ktd * x_td)                   
            + ( self.Kde * x_cde) + (-self.Kdd * x_cdd) + ( self.Kte * x_cte) + (-self.Ktd * x_ctd)               
            + (-self.Cde * xdot_de) + ( self.Cdd * xdot_dd) + (-self.Cte * xdot_te) + ( self.Ctd * xdot_td)       
            + ( self.Cde * xdot_cde) + (-self.Cdd * xdot_cdd) + ( self.Cte * xdot_cte) + (-self.Ctd * xdot_ctd))  
            - (self.Kt * phi + self.Cphi * phidot) / self.Itorc                                                   
            )

        theta_acc = (1 / self.Iflex) * (
            + self.Ld * ( self.Kde * x_de + self.Kdd * x_dd - self.Kde * x_cde - self.Kdd * x_cdd )
            - self.Lt * ( self.Kte * x_te + self.Ktd * x_td - self.Kte * x_cte - self.Ktd * x_ctd )
            + self.Ld * ( self.Cde * xdot_de + self.Cdd * xdot_dd - self.Cde * xdot_cde - self.Cdd * xdot_cdd )
            - self.Lt * ( self.Cte * xdot_te + self.Ctd * xdot_td - self.Cte * xdot_cte - self.Ctd * xdot_ctd )
            - (self.Kf * theta + self.Ctheta * thetadot)
        )

        " ------------------------------- Metamodelo de Dinâmica ------------------------------- "
        
        Fz_total_state = Fz_de_curr + Fz_dd_curr + Fz_te_curr + Fz_td_curr + 1e-6

        Fx_de = self.Fx_total * (Fz_de_curr / Fz_total_state)
        Fx_dd = self.Fx_total * (Fz_dd_curr / Fz_total_state)
        Fx_te = self.Fx_total * (Fz_te_curr / Fz_total_state)
        Fx_td = self.Fx_total * (Fz_td_curr / Fz_total_state)

        Fy_de = self.Fy_total * (Fz_de_curr / Fz_total_state)
        Fy_dd = self.Fy_total * (Fz_dd_curr / Fz_total_state)
        Fy_te = self.Fy_total * (Fz_te_curr / Fz_total_state)
        Fy_td = self.Fy_total * (Fz_td_curr / Fz_total_state)

        Fz_dd_target, Fz_de_target, Fz_td_target, Fz_te_target = self.dynamics.load_transfer(
                                    Fz_static_front=self.Fz_static_front, Fz_static_rear=self.Fz_static_rear,
                                    Fx_de=Fx_de, Fx_dd=Fx_dd, Fx_te=Fx_te, Fx_td=Fx_td,
                                    Fy_de=Fy_de, Fy_dd=Fy_dd, Fy_te=Fy_te, Fy_td=Fy_td)
        
        tau = 0.05 
        dFz_de = (Fz_de_target - Fz_de_curr) / tau
        dFz_dd = (Fz_dd_target - Fz_dd_curr) / tau
        dFz_te = (Fz_te_target - Fz_te_curr) / tau
        dFz_td = (Fz_td_target - Fz_td_curr) / tau

        yawddot = self.dynamics.yaw_acc(Fx_de=Fx_de, Fx_dd=Fx_dd, Fx_te=Fx_te, Fx_td=Fx_td, Fy_de=Fy_de, Fy_dd=Fy_dd, Fy_te=Fy_te, Fy_td=Fy_td)
        rollddot, _ = self.dynamics.roll_acc(Fy_de=Fy_de, Fy_dd=Fy_dd, Fy_te=Fy_te, Fy_td=Fy_td, chassis_F_de=F_de, chassis_F_dd=F_dd, chassis_F_te=F_te, chassis_F_td=F_td)
        pitchddot, _ = self.dynamics.pitch_acc(Fx_de=Fx_de, Fx_dd=Fx_dd, Fx_te=Fx_te, Fx_td=Fx_td, chassis_F_de=F_de, chassis_F_dd=F_dd, chassis_F_te=F_te, chassis_F_td=F_td)

        x_acc_car = ((Fx_de + Fx_dd + Fx_te + Fx_td) * np.cos(yaw) + (Fy_de + Fy_dd + Fy_te + Fy_td) * np.sin(yaw))/ self.Mc
        y_acc_car = ((Fy_de + Fy_dd + Fy_te + Fy_td) * np.cos(yaw) + (Fx_de + Fx_dd + Fx_te + Fx_td) * np.sin(yaw))/ self.Mc

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
            dFz_de, dFz_dd, dFz_te, dFz_td, 
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

def get_simulation_time(t_start, t_end, dt):
    t_span = (t_start, t_end)
    t_eval = np.arange(t_start, t_end+dt, dt)
    return t_span, t_eval

def run_simulation():
    
    t_span, t_eval = get_simulation_time(t_start=0.0, t_end=10.0, dt=0.001)

    Ld, Lt, W = 1.4, 1.1, 0.5

    altura_degrau_d = 1 
    altura_degrau_e = 1
    tempo_inicio_direito = 1.0
    tempo_inicio_esquerdo = 1.0

    Forca_Longitudinal_Total = 2000 
    Forca_Lateral_Total = 0      

    chassis = FullModel(
        Mc=280, Mst=25, Msd=20, Distribution=0.5,
        Kte=5e5, Ktd=5e5, Kde=5e5, Kdd=5e5,
        Kpte=2e4, Kptd=2e4, Kpde=2e4, Kpdd=2e4,
        Kf=3.01e5, Kt=1.6e5, 
        Cte=5e3, Ctd=5e3, Cde=3e4, Cdd=3e3,
        Cphi=2e4, Ctheta=2e4,
        Iflex=5e2, Itorc=5e2,
        W=W, Lt=Lt, Ld=Ld, hcg=0.3,
        tire_coef=1.45, params=(0.3336564873588197, 1.627, 1.0, 931.405, 366.493),
        Fx_total=Forca_Longitudinal_Total, 
        Fy_total=Forca_Lateral_Total,
        Ix=1e3, Iz=1e3, Iy=1e3
    )

    funcoes_entrada = chassis.gerar_funcoes_entrada(tempo_inicio_direito, tempo_inicio_esquerdo,
                                                    altura_degrau_d, altura_degrau_e)
    
    Fz_de0 = chassis.Fz_static_front / 2
    Fz_dd0 = chassis.Fz_static_front / 2
    Fz_te0 = chassis.Fz_static_rear / 2
    Fz_td0 = chassis.Fz_static_rear / 2

    y0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Fz_de0, Fz_dd0, Fz_te0, Fz_td0,0,0,0,0]
    sol = solve_ivp(fun=lambda t,y: chassis.ode_total(t,y,funcoes_entrada),
                t_span=t_span, y0=y0, t_eval=t_eval, method='LSODA')
    
    t = sol.t 

    solucao_roda_de = sol.y[0]    
    solucao_roda_dd = sol.y[2]    
    solucao_roda_te = sol.y[4]    
    solucao_roda_td = sol.y[6]    
    solucao_torcao  = sol.y[8]    
    solucao_flexao  = sol.y[10]   
    solucao_carro   = sol.y[12]   

    solucao_yaw     = sol.y[14]   
    solucao_roll    = sol.y[18]   
    solucao_pitch   = sol.y[16]   

    solucao_Fz_de = sol.y[20]
    solucao_Fz_dd = sol.y[21]
    solucao_Fz_te = sol.y[22]
    solucao_Fz_td = sol.y[23]

    solucao_x       = sol.y[24]
    solucao_y       = sol.y[25]
    solucao_vx      = sol.y[26]
    solucao_vy      = sol.y[27]
    
    plot_deslocamento_rodas = True
    plot_dinamica_chassi = True
    plot_cargas_fz = True

    if plot_deslocamento_rodas:
        plt.figure()
        plt.plot(t, solucao_roda_de, label="Roda Dianteira Esquerda")
        plt.plot(t, solucao_roda_dd, label="Roda Dianteira Direita")
        plt.plot(t, solucao_roda_te, label="Roda Traseira Esquerda")
        plt.plot(t, solucao_roda_td, label="Roda Traseira Direita")
        plt.legend()
        plt.xlabel("Tempo [s]")
        plt.ylabel("Deslocamento [m]")
        plt.title("Deslocamento Vertical das Rodas")
        plt.grid(True)

    if plot_dinamica_chassi:
        plt.figure()
        plt.plot(t, solucao_torcao, label="Torção do Chassi")
        plt.plot(t, solucao_flexao, label="Flexão do Chassi")
        plt.legend()
        plt.xlabel("Tempo [s]")
        plt.ylabel("Deslocamento [m ou rad]")
        plt.title("Dinâmica do Chassi")
        plt.grid(True)

    if plot_cargas_fz:
        plt.figure()
        plt.plot(t, solucao_Fz_de, label="Fz Dianteira Esquerda")
        plt.plot(t, solucao_Fz_dd, label="Fz Dianteira Direita")
        plt.plot(t, solucao_Fz_te, label="Fz Traseira Esquerda")
        plt.plot(t, solucao_Fz_td, label="Fz Traseira Direita")
        plt.xlabel("Tempo [s]")
        plt.ylabel("Carga Vertical [N]")
        plt.title("Resposta de Carga Vertical (Fz) nos Pneus")
        plt.legend()
        plt.grid(True)

    plt.show()

if __name__ == "__main__":
    run_simulation()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import functools
from enum import Enum

# =============================================================================
# Modo de operação
# =============================================================================
class DriveMode(Enum):
    ACCELERATION = 1   # só traseiras tracionam
    BRAKING = 2        # quatro rodas freiam
    CORNERING = 3      # dianteiras full, traseiras 50% da capacidade lateral

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
        beta = 0.0005  # parâmetro de sensibilidade (ajuste fino)
        D = self.tire_friction_coef * Fz * (np.log(1 + beta * Fz) / (beta * Fz))
        Bx = Cs / (Cx * D) if (Cx * D) != 0 else 0.0
        By = Cs / (Cy * D) if (Cy * D) != 0 else 0.0
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
        roll_moment_total  = (Fy_de + Fy_dd + Fy_te + Fy_td) * self.hcg

        wheelbase = self.Ld + self.Lt
        track_width = self.W * 2

        pitch_load_delta = pitch_moment_total / wheelbase if wheelbase != 0 else 0.0
        roll_load_delta  = roll_moment_total  / track_width if track_width != 0 else 0.0

        Tire_Fz_dd = Fz_static_front/2 + roll_load_delta - pitch_load_delta
        Tire_Fz_de = Fz_static_front/2 - roll_load_delta - pitch_load_delta
        Tire_Fz_td = Fz_static_rear/2  + roll_load_delta + pitch_load_delta
        Tire_Fz_te = Fz_static_rear/2  - roll_load_delta + pitch_load_delta

        # Evita valores negativos (numericamente pode acontecer)
        Tire_Fz_dd = max(Tire_Fz_dd, 0.0)
        Tire_Fz_de = max(Tire_Fz_de, 0.0)
        Tire_Fz_td = max(Tire_Fz_td, 0.0)
        Tire_Fz_te = max(Tire_Fz_te, 0.0)

        return Tire_Fz_dd, Tire_Fz_de, Tire_Fz_td, Tire_Fz_te

class FullModel:
    def __init__(self,
        Distribution: float, Kds: float, Kts: float, Kpds: float, Kpts: float, 
        Kf: float, Kt: float, Cds: float, Cts: float,
        W: float, Lt: float, Ld: float,
        Mc=280, Mst=25, Msd=20,
        Cphi=2e4, Ctheta=2e4,
        Iflex=5e2, Itorc=5e2, hcg=0.3,
        tire_coef=1.45,params=(0.3336564873588197, 1.627, 1.0, 931.405, 366.493),
        Fx_total=0, Fy_total=0,
        Ix=1e3, Iz=1e3, Iy=1e3,
        mode=DriveMode.ACCELERATION
    ):

        # Massas e Parâmetros
        self.Mc = Mc
        self.Mst_unsprung = Mst
        self.Msd_unsprung = Msd
        self.Distribution = Distribution
        self.Fz_static_rear  = 9.81 * Mc * Distribution/2
        self.Fz_static_front = 9.81 * Mc * (1 - Distribution)/2

        self.Kde = self.Kdd = Kds
        self.Kte = self.Ktd = Kts
        self.Kpde = self.Kpdd = Kpds
        self.Kpte = self.Kptd = Kpts
        self.Kf, self.Kt = Kf, Kt
        self.Cde = self.Cdd = Cds
        self.Cte = self.Ctd = Cts
        self.Cphi, self.Ctheta = Cphi, Ctheta
        self.Iflex, self.Itorc = Iflex, Itorc
        self.W, self.Lt, self.Ld = W, Lt, Ld
        self.Fx_total, self.Fy_total = Fx_total, Fy_total

        self.tire = TireModel(tire_friction_coef=tire_coef, params=params)
        self.dynamics = VehicleDynamics(mass=Mc, Iz=Iz, Ix=Ix, Iy=Iy, Ld=Ld, Lt=Lt, W=W, hcg=hcg)

        # Modo de operação e tracking de máximas
        self.mode = mode
        self.max_ax = 0.0
        self.max_ay = 0.0
        self.max_Fx_total = 0.0
        self.max_Fy_total = 0.0

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
        x_cde = Xc + self.Ld * np.sin(theta) + self.W * np.sin(phi)
        xdot_cde = Xc_dot + self.Ld * np.cos(theta) * thetadot + self.W * np.cos(phi) * phidot

        x_cdd = Xc + self.Ld * np.sin(theta) - self.W * np.sin(phi)
        xdot_cdd = Xc_dot + self.Ld * np.cos(theta) * thetadot - self.W * np.cos(phi) * phidot

        x_cte = Xc - self.Lt * np.sin(theta) + self.W * np.sin(phi)
        xdot_cte = Xc_dot - self.Lt * np.cos(theta) * thetadot + self.W * np.cos(phi) * phidot

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

        " ------------------------------- Metamodelo de Dinâmica com correção de limites ------------------------------- "
        # 1) Limites com Fz corrente (estado) via Pacejka.D
        _, _, _, _, _, D_de_curr = self.tire.pacejka_params(Fz_de_curr)
        _, _, _, _, _, D_dd_curr = self.tire.pacejka_params(Fz_dd_curr)
        _, _, _, _, _, D_te_curr = self.tire.pacejka_params(Fz_te_curr)
        _, _, _, _, _, D_td_curr = self.tire.pacejka_params(Fz_td_curr)

        Fx_lim_de_1 = D_de_curr
        Fx_lim_dd_1 = D_dd_curr
        Fx_lim_te_1 = D_te_curr
        Fx_lim_td_1 = D_td_curr

        Fy_lim_de_1 = D_de_curr
        Fy_lim_dd_1 = D_dd_curr
        Fy_lim_te_1 = 0.5 * D_te_curr
        Fy_lim_td_1 = 0.5 * D_td_curr

        # 1a) Forças provisórias com limites correntes
        Fx_de = Fx_dd = Fx_te = Fx_td = 0.0
        Fy_de = Fy_dd = Fy_te = Fy_td = 0.0

        if self.mode == DriveMode.ACCELERATION:
            Fx_request_total = self.Fx_total
            total_rear_Fz = Fz_te_curr + Fz_td_curr + 1e-9
            Fx_te_req = Fx_request_total * (Fz_te_curr / total_rear_Fz)
            Fx_td_req = Fx_request_total * (Fz_td_curr / total_rear_Fz)
            Fx_te = np.clip(Fx_te_req, 0.0, Fx_lim_te_1)
            Fx_td = np.clip(Fx_td_req, 0.0, Fx_lim_td_1)

        elif self.mode == DriveMode.BRAKING:
            Fx_request_total = -abs(self.Fx_total)
            total_Fz = Fz_de_curr + Fz_dd_curr + Fz_te_curr + Fz_td_curr + 1e-9
            Fx_de_req = Fx_request_total * (Fz_de_curr / total_Fz)
            Fx_dd_req = Fx_request_total * (Fz_dd_curr / total_Fz)
            Fx_te_req = Fx_request_total * (Fz_te_curr / total_Fz)
            Fx_td_req = Fx_request_total * (Fz_td_curr / total_Fz)
            Fx_de = np.clip(Fx_de_req, -Fx_lim_de_1, 0.0)
            Fx_dd = np.clip(Fx_dd_req, -Fx_lim_dd_1, 0.0)
            Fx_te = np.clip(Fx_te_req, -Fx_lim_te_1, 0.0)
            Fx_td = np.clip(Fx_td_req, -Fx_lim_td_1, 0.0)

        elif self.mode == DriveMode.CORNERING:
            Fy_request_total = self.Fy_total
            cap_front = Fy_lim_de_1 + Fy_lim_dd_1 + 1e-9
            cap_rear  = Fy_lim_te_1 + Fy_lim_td_1 + 1e-9
            cap_total = cap_front + cap_rear
            Fy_front_req = Fy_request_total * (cap_front / cap_total)
            Fy_rear_req  = Fy_request_total * (cap_rear  / cap_total)
            front_Fz = Fz_de_curr + Fz_dd_curr + 1e-9
            rear_Fz  = Fz_te_curr + Fz_td_curr + 1e-9
            Fy_de_req = Fy_front_req * (Fz_de_curr / front_Fz)
            Fy_dd_req = Fy_front_req * (Fz_dd_curr / front_Fz)
            Fy_te_req = Fy_rear_req  * (Fz_te_curr / rear_Fz)
            Fy_td_req = Fy_rear_req  * (Fz_td_curr / rear_Fz)
            Fy_de = np.clip(Fy_de_req, -Fy_lim_de_1,  Fy_lim_de_1)
            Fy_dd = np.clip(Fy_dd_req, -Fy_lim_dd_1,  Fy_lim_dd_1)
            Fy_te = np.clip(Fy_te_req, -Fy_lim_te_1,  Fy_lim_te_1)
            Fy_td = np.clip(Fy_td_req, -Fy_lim_td_1,  Fy_lim_td_1)

        # 2) Transferência de carga para obter Fz_target (com forças provisórias)
        Fz_dd_target, Fz_de_target, Fz_td_target, Fz_te_target = self.dynamics.load_transfer(
            Fz_static_front=self.Fz_static_front, Fz_static_rear=self.Fz_static_rear,
            Fx_de=Fx_de, Fx_dd=Fx_dd, Fx_te=Fx_te, Fx_td=Fx_td,
            Fy_de=Fy_de, Fy_dd=Fy_dd, Fy_te=Fy_te, Fy_td=Fy_td)

        # 3) Limites via Pacejka usando Fz_target (correção para usar capacidade instantânea)
        _, _, _, _, _, D_de_tgt = self.tire.pacejka_params(Fz_de_target)
        _, _, _, _, _, D_dd_tgt = self.tire.pacejka_params(Fz_dd_target)
        _, _, _, _, _, D_te_tgt = self.tire.pacejka_params(Fz_te_target)
        _, _, _, _, _, D_td_tgt = self.tire.pacejka_params(Fz_td_target)

        Fx_lim_de_2 = D_de_tgt
        Fx_lim_dd_2 = D_dd_tgt
        Fx_lim_te_2 = D_te_tgt
        Fx_lim_td_2 = D_td_tgt

        Fy_lim_de_2 = D_de_tgt
        Fy_lim_dd_2 = D_dd_tgt
        Fy_lim_te_2 = 0.5 * D_te_tgt
        Fy_lim_td_2 = 0.5 * D_td_tgt

        # 3a) Recalcula forças com limites "instantâneos"
        if self.mode == DriveMode.ACCELERATION:
            Fx_request_total = self.Fx_total
            total_rear_Fz_tgt = Fz_te_target + Fz_td_target + 1e-9
            Fx_te_req = Fx_request_total * (Fz_te_target / total_rear_Fz_tgt)
            Fx_td_req = Fx_request_total * (Fz_td_target / total_rear_Fz_tgt)
            Fx_te = np.clip(Fx_te_req, 0.0, Fx_lim_te_2)
            Fx_td = np.clip(Fx_td_req, 0.0, Fx_lim_td_2)
            # dianteiras não tracionam
            Fx_de = 0.0
            Fx_dd = 0.0

        elif self.mode == DriveMode.BRAKING:
            Fx_request_total = -abs(self.Fx_total)
            total_Fz_tgt = Fz_de_target + Fz_dd_target + Fz_te_target + Fz_td_target + 1e-9
            Fx_de_req = Fx_request_total * (Fz_de_target / total_Fz_tgt)
            Fx_dd_req = Fx_request_total * (Fz_dd_target / total_Fz_tgt)
            Fx_te_req = Fx_request_total * (Fz_te_target / total_Fz_tgt)
            Fx_td_req = Fx_request_total * (Fz_td_target / total_Fz_tgt)
            Fx_de = np.clip(Fx_de_req, -Fx_lim_de_2, 0.0)
            Fx_dd = np.clip(Fx_dd_req, -Fx_lim_dd_2, 0.0)
            Fx_te = np.clip(Fx_te_req, -Fx_lim_te_2, 0.0)
            Fx_td = np.clip(Fx_td_req, -Fx_lim_td_2, 0.0)

        elif self.mode == DriveMode.CORNERING:
            Fy_request_total = self.Fy_total
            cap_front_2 = Fy_lim_de_2 + Fy_lim_dd_2 + 1e-9
            cap_rear_2  = Fy_lim_te_2 + Fy_lim_td_2 + 1e-9
            cap_total_2 = cap_front_2 + cap_rear_2
            Fy_front_req = Fy_request_total * (cap_front_2 / cap_total_2)
            Fy_rear_req  = Fy_request_total * (cap_rear_2  / cap_total_2)
            front_Fz_tgt = Fz_de_target + Fz_dd_target + 1e-9
            rear_Fz_tgt  = Fz_te_target + Fz_td_target + 1e-9
            Fy_de_req = Fy_front_req * (Fz_de_target / front_Fz_tgt)
            Fy_dd_req = Fy_front_req * (Fz_dd_target / front_Fz_tgt)
            Fy_te_req = Fy_rear_req  * (Fz_te_target / rear_Fz_tgt)
            Fy_td_req = Fy_rear_req  * (Fz_td_target / rear_Fz_tgt)
            Fy_de = np.clip(Fy_de_req, -Fy_lim_de_2,  Fy_lim_de_2)
            Fy_dd = np.clip(Fy_dd_req, -Fy_lim_dd_2,  Fy_lim_dd_2)
            Fy_te = np.clip(Fy_te_req, -Fy_lim_te_2,  Fy_lim_te_2)
            Fy_td = np.clip(Fy_td_req, -Fy_lim_td_2,  Fy_lim_td_2)
            # sem força longitudinal em curva pura
            Fx_de = Fx_dd = Fx_te = Fx_td = 0.0

        # 4) Atualiza Fz (estado) com dinâmica primeiro ordem rumo ao target
        tau = 0.05
        dFz_de = (Fz_de_target - Fz_de_curr) / tau
        dFz_dd = (Fz_dd_target - Fz_dd_curr) / tau
        dFz_te = (Fz_te_target - Fz_te_curr) / tau
        dFz_td = (Fz_td_target - Fz_td_curr) / tau

        # 5) Dinâmica global
        yawddot = self.dynamics.yaw_acc(Fx_de=Fx_de, Fx_dd=Fx_dd, Fx_te=Fx_te, Fx_td=Fx_td,
                                        Fy_de=Fy_de, Fy_dd=Fy_dd, Fy_te=Fy_te, Fy_td=Fy_td)
        rollddot, _ = self.dynamics.roll_acc(Fy_de=Fy_de, Fy_dd=Fy_dd, Fy_te=Fy_te, Fy_td=Fy_td,
                                             chassis_F_de=F_de, chassis_F_dd=F_dd, chassis_F_te=F_te, chassis_F_td=F_td)
        pitchddot, _ = self.dynamics.pitch_acc(Fx_de=Fx_de, Fx_dd=Fx_dd, Fx_te=Fx_te, Fx_td=Fx_td,
                                               chassis_F_de=F_de, chassis_F_dd=F_dd, chassis_F_te=F_te, chassis_F_td=F_td)

        x_acc_car = ((Fx_de + Fx_dd + Fx_te + Fx_td) * np.cos(yaw) + (Fy_de + Fy_dd + Fy_te + Fy_td) * np.sin(yaw))/ self.Mc
        y_acc_car = ((Fy_de + Fy_dd + Fy_te + Fy_td) * np.cos(yaw) + (Fx_de + Fx_dd + Fx_te + Fx_td) * np.sin(yaw))/ self.Mc

        # Tracking das máximas (magnitude) e forças correspondentes
        total_Fx = abs(Fx_de + Fx_dd + Fx_te + Fx_td)
        total_Fy = abs(Fy_de + Fy_dd + Fy_te + Fy_td)
        self.max_ax = max(self.max_ax, abs(x_acc_car))
        self.max_ay = max(self.max_ay, abs(y_acc_car))
        self.max_Fx_total = max(self.max_Fx_total, total_Fx)
        self.max_Fy_total = max(self.max_Fy_total, total_Fy)

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

    def __get_initial_state(self):
        Fz_de0 = self.Fz_static_front / 2
        Fz_dd0 = self.Fz_static_front / 2
        Fz_te0 = self.Fz_static_rear / 2
        Fz_td0 = self.Fz_static_rear / 2
        y0 = [0,0,0,0,0,0,0,0,
            0,0, 0,0, 0,0, 0,0, 0,0, 0,0,
            Fz_de0, Fz_dd0, Fz_te0, Fz_td0,
            0,0, 0,0]
        return y0

    def __get_max_acceleration_for_mode(self, mode: DriveMode, test_force: float = 1e6, total_time: float = 10, dt: float = 0.001) -> float:
        t_span = (0, total_time)
        t_eval = np.arange(0, total_time+dt, dt)

        road_funcs = self.gerar_funcoes_entrada(1, 1, 1, 1)
    
        self.mode = mode
        self.Fx_total = test_force if mode != DriveMode.CORNERING else 0.0
        self.Fy_total = test_force if mode == DriveMode.CORNERING else 0.0
        self.max_ax = self.max_ay = 0.0
        self.max_Fx_total = self.max_Fy_total = 0.0

        y0 = self.__get_initial_state()
        solve_ivp(fun=lambda t,y: self.ode_total(t,y,road_funcs),
                t_span=t_span, y0=y0, t_eval=t_eval, method='LSODA')
        
        return self.max_ay if mode == DriveMode.CORNERING else self.max_ax

    def get_acceleration_limits(self) -> tuple[float, float, float]:
        return self.__get_max_acceleration_for_mode(DriveMode.ACCELERATION), \
               self.__get_max_acceleration_for_mode(DriveMode.BRAKING), \
               self.__get_max_acceleration_for_mode(DriveMode.CORNERING)


               
# =============================================================================
# Utilitários de simulação
# =============================================================================
def get_simulation_time(t_start, t_end, dt):
    t_span = (t_start, t_end)
    t_eval = np.arange(t_start, t_end+dt, dt)
    return t_span, t_eval

def get_initial_state(chassis):
    Fz_de0 = chassis.Fz_static_front / 2
    Fz_dd0 = chassis.Fz_static_front / 2
    Fz_te0 = chassis.Fz_static_rear / 2
    Fz_td0 = chassis.Fz_static_rear / 2
    y0 = [0,0,0,0,0,0,0,0,
          0,0, 0,0, 0,0, 0,0, 0,0, 0,0,
          Fz_de0, Fz_dd0, Fz_te0, Fz_td0,
          0,0, 0,0]
    return y0

# =============================================================================
# Funções para máximas de aceleração
# =============================================================================
def get_max_longitudinal_acceleration(chassis, Fx_request, t_span, t_eval, road_funcs):
    chassis.mode = DriveMode.ACCELERATION
    chassis.Fx_total = Fx_request
    chassis.Fy_total = 0.0
    chassis.max_ax = chassis.max_ay = 0.0
    chassis.max_Fx_total = chassis.max_Fy_total = 0.0

    y0 = get_initial_state(chassis)
    solve_ivp(fun=lambda t,y: chassis.ode_total(t,y,road_funcs),
              t_span=t_span, y0=y0, t_eval=t_eval, method='LSODA')
    return chassis.max_ax, chassis.max_Fx_total

def get_max_braking_acceleration(chassis, Fx_request, t_span, t_eval, road_funcs):
    chassis.mode = DriveMode.BRAKING
    chassis.Fx_total = abs(Fx_request)   # negativo internamente
    chassis.Fy_total = 0.0
    chassis.max_ax = chassis.max_ay = 0.0
    chassis.max_Fx_total = chassis.max_Fy_total = 0.0

    y0 = get_initial_state(chassis)
    solve_ivp(fun=lambda t,y: chassis.ode_total(t,y,road_funcs),
              t_span=t_span, y0=y0, t_eval=t_eval, method='LSODA')
    return chassis.max_ax, chassis.max_Fx_total

def get_max_lateral_acceleration(chassis, Fy_request, t_span, t_eval, road_funcs):
    chassis.mode = DriveMode.CORNERING
    chassis.Fx_total = 0.0
    chassis.Fy_total = Fy_request
    chassis.max_ax = chassis.max_ay = 0.0
    chassis.max_Fx_total = chassis.max_Fy_total = 0.0

    y0 = get_initial_state(chassis)
    solve_ivp(fun=lambda t,y: chassis.ode_total(t,y,road_funcs),
              t_span=t_span, y0=y0, t_eval=t_eval, method='LSODA')
    return chassis.max_ay, chassis.max_Fy_total

# =============================================================================
# Execução e gráficos originais
# =============================================================================
def run_simulation():
    t_span, t_eval = get_simulation_time(t_start=0.0, t_end=10.0, dt=0.001)

    Ld, Lt, W = 1.4, 1.1, 1

    altura_degrau_d = 1
    altura_degrau_e = 1
    tempo_inicio_direito = 1.0
    tempo_inicio_esquerdo = 1.0

    # Força "pedido" para simulações de máximas (grandes para forçar saturação por roda)
    Fx_req_accel   = 1e6
    Fx_req_braking = 1e6
    Fy_req_corner  = 1e6

    # Instancia o chassis
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
        Fx_total=0.0, Fy_total=0.0,
        Ix=1e3, Iz=1e3, Iy=1e3,
        mode=DriveMode.ACCELERATION
    )

    funcoes_entrada = chassis.gerar_funcoes_entrada(tempo_inicio_direito, tempo_inicio_esquerdo,
                                                    altura_degrau_d, altura_degrau_e)

    # 1) Medição de máximas de aceleração
    a_long_max, F_long_max = get_max_longitudinal_acceleration(chassis, Fx_req_accel,   t_span, t_eval, funcoes_entrada)
    a_brake_max, F_brake_max = get_max_braking_acceleration(chassis,     Fx_req_braking, t_span, t_eval, funcoes_entrada)
    a_lat_max,   F_lat_max   = get_max_lateral_acceleration(chassis,     Fy_req_corner,  t_span, t_eval, funcoes_entrada)

    print(f"Aceleração longitudinal máxima: {a_long_max:.3f} m/s^2")
    print(f"Força longitudinal máxima (somatório rodas): {F_long_max:.1f} N")
    print(f"Desaceleração máxima (frenagem): {a_brake_max:.3f} m/s^2")
    print(f"Força de frenagem máxima (somatório rodas): {F_brake_max:.1f} N")
    print(f"Aceleração lateral máxima: {a_lat_max:.3f} m/s^2")
    print(f"Força lateral máxima (somatório rodas): {F_lat_max:.1f} N")

    # 2) Simulação original para gráficos (caso de aceleração moderada)
    chassis.mode = DriveMode.ACCELERATION
    chassis.Fx_total = 2000
    chassis.Fy_total = 0

    y0 = get_initial_state(chassis)
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

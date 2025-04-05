import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

class control:

    def __init__(self, tire_Sa=None, front_tire_Ls=None, rear_tire_Ls=None, tire_friction_coef=None, tire_Ca=0, 
                params=None, mass=None, weight_distribution=None, rear_Ca=None, front_Ca=None, 
                Ld=None, Lt=None, W2=None, yaw_inertia=None, roll_inertia=None, pitch_inertia=None, hcg=None):

        self.tire_Sa = tire_Sa  # Ângulo de deslizamento lateral do pneu [rad]
        self.rear_tire_Sa = tire_Sa / 10
        self.front_tire_Ls = front_tire_Ls  # Escorregamento longitudinal frontal [adimensional]
        self.rear_tire_Ls = rear_tire_Ls    # Escorregamento longitudinal traseiro [adimensional]
        self.tire_friction_coef = tire_friction_coef  # Coeficiente de atrito entre pneu e pista
        self.tire_Ca = tire_Ca  # Ângulo de camber do pneu
        self.params = params  # Parâmetros de Pacejka (E, Cy, Cx, c1, c2)
        self.mass = mass      # Massa do veículo (kg)
        self.weight_distribution = weight_distribution  # Distribuição de peso na traseira (adimensional)
        self.Ld = Ld  # Distância do CG para o eixo dianteiro (m)
        self.Lt = Lt  # Distância do CG para o eixo traseiro (m)
        self.W2 = W2  # Metade da largura do veículo (m)
        self.hcg = hcg  # Altura do centro de gravidade (m)

        self.yaw_inertia = yaw_inertia    # Inércia em yaw (kg.m^2)
        self.roll_inertia = roll_inertia  # Inércia em rolagem (kg.m^2)
        self.pitch_inertia = pitch_inertia  # Inércia em pitch (kg.m^2)

        rear_load = 9.81 * mass * weight_distribution  # Carga no eixo traseiro (N)
        front_load = 9.81 * mass * (1 - weight_distribution)  # Carga no eixo dianteiro (N)

        self.rear_Ca = rear_Ca
        self.front_Ca = front_Ca
        
        self.tire_Fz_de = front_load / 2 
        self.tire_Fz_dd = front_load / 2  
        self.tire_Fz_te = rear_load / 2
        self.tire_Fz_td = rear_load / 2

    def pacejka_params(self, tire_Fz):
        # Desembalando os parâmetros de Pacejka
        E, Cy, Cx, c1, c2 = self.params
        # Calculando parâmetros intermediários
        Cs = c1 * np.sin(2 * np.arctan(tire_Fz / c2))
        D = self.tire_friction_coef * tire_Fz
        Bx = Cs / (Cx * D)
        By = Cs / (Cy * D)
        return E, Cy, Cx, Bx, By, D

    def lateral_force(self, tire_Fz, tire_Ca):
        # Desembalando os parâmetros de Pacejka
        E, Cy, Cx, Bx, By, D = control.pacejka_params(self, tire_Fz)
        # Calculando a força lateral do pneu
        tire_lateral_force = D * np.sin(Cy * np.arctan(By * self.tire_Sa - E * (By * self.tire_Sa - np.arctan(By * self.tire_Sa))))
        # Calculando a força de camber
        camber_thrust = D / 2 * np.sin(Cy * np.arctan(By * tire_Ca))
        return tire_lateral_force + camber_thrust
    
    def longitudinal_force(self, tire_Fz, front):
        if front:
            tire_Ls = self.front_tire_Ls
        else:
            tire_Ls = self.rear_tire_Ls
        # Desembalando os parâmetros de Pacejka
        E, Cy, Cx, Bx, By, D = control.pacejka_params(self, tire_Fz)
        # Calculando a força longitudinal do pneu
        tire_longitudinal_force = D * np.sin(Cx * np.arctan(9 * Bx * tire_Ls - E * (Bx * tire_Ls - np.arctan(Bx * tire_Ls))))
        return tire_longitudinal_force
    
    def yaw(self):
        tire_y_de = control.lateral_force(self, tire_Fz=self.tire_Fz_de, tire_Ca=self.front_Ca)
        tire_y_dd = control.lateral_force(self, tire_Fz=self.tire_Fz_dd, tire_Ca=self.front_Ca)
        tire_y_te = control.lateral_force(self, tire_Fz=self.tire_Fz_te, tire_Ca=self.rear_Ca)
        tire_y_td = control.lateral_force(self, tire_Fz=self.tire_Fz_td, tire_Ca=self.rear_Ca)

        tire_x_de = control.longitudinal_force(self, tire_Fz=self.tire_Fz_de, front=True)
        tire_x_dd = control.longitudinal_force(self, tire_Fz=self.tire_Fz_dd, front=True)
        tire_x_te = control.longitudinal_force(self, tire_Fz=self.tire_Fz_te, front=False)
        tire_x_td = control.longitudinal_force(self, tire_Fz=self.tire_Fz_td, front=False)

        lateral_moment = (tire_y_de + tire_y_dd) * self.Ld - (tire_y_te + tire_y_td) * self.Lt
        longitudinal_moment = (tire_x_de - tire_x_dd + tire_x_te - tire_x_td) * self.W2

        ddot_yaw = (lateral_moment + longitudinal_moment) / self.yaw_inertia
        return ddot_yaw
    
    def roll_moment(self):
        tire_y_de = control.lateral_force(self, tire_Fz=self.tire_Fz_de, tire_Ca=self.front_Ca)
        tire_y_dd = control.lateral_force(self, tire_Fz=self.tire_Fz_dd, tire_Ca=self.front_Ca)
        tire_y_te = control.lateral_force(self, tire_Fz=self.tire_Fz_te, tire_Ca=self.rear_Ca)
        tire_y_td = control.lateral_force(self, tire_Fz=self.tire_Fz_td, tire_Ca=self.rear_Ca)

        roll_moment = (tire_y_de + tire_y_dd + tire_y_te + tire_y_td) * self.hcg
        ddot_roll = roll_moment / self.roll_inertia
        return ddot_roll, roll_moment
    
    def pitch_moment(self):
        tire_x_de = control.longitudinal_force(self, tire_Fz=self.tire_Fz_de, front=True)
        tire_x_dd = control.longitudinal_force(self, tire_Fz=self.tire_Fz_dd, front=True)
        tire_x_te = control.longitudinal_force(self, tire_Fz=self.tire_Fz_te, front=False)
        tire_x_td = control.longitudinal_force(self, tire_Fz=self.tire_Fz_td, front=False)

        pitch_moment = (tire_x_de + tire_x_dd + tire_x_te + tire_x_td) * self.hcg
        ddot_pitch = pitch_moment / self.pitch_inertia
        return ddot_pitch, pitch_moment

    def integrate_motion(t, acceleration):
        """
        Integra a aceleração para obter velocidade e deslocamento.
        t: vetor de tempo
        acceleration: vetor de aceleração
        Retorna (velocity, displacement) com condições iniciais zero.
        """
        velocity = cumtrapz(acceleration, t, initial=0)
        displacement = cumtrapz(velocity, t, initial=0)
        return velocity, displacement
    
    def load_transfer(self):
        pitch_moment = control.pitch_moment(self)[1]
        roll_moment = control.roll_moment(self)[1]

        pitch_load = pitch_moment/(self.Lt + self.Ld)
        roll_load = roll_moment/(2*self.W2)

        self.tire_Fz_dd = self.tire_Fz_dd - roll_load - pitch_load
        self.tire_Fz_de = self.tire_Fz_de + roll_load - pitch_load
        self.tire_Fz_td = self.tire_Fz_td - roll_load + pitch_load
        self.tire_Fz_te = self.tire_Fz_te + roll_load + pitch_load
        
    
    def model_input():
        # Parâmetros fornecidos
        E = 0.3336564873588197
        C_y = 1.627
        C_x = 1.0
        c1 = 931.405
        c2 = 366.493

        Ld = 1.2    # Distância do CG para dianteiro [m]
        Lt = 1.2    # Distância do CG para traseiro [m]
        W2 = 0.45   # Metade da largura do carro [m]
        H_cg = 0.3  # Altura do centro de gravidade [m]
        I_psi = 1e2   # Inércia em Yaw [kg.m^2]
        I_rho = 1e2   # Inércia em Rolagem [kg.m^2]
        I_beta = 1e2  # Inércia em Pitch [kg.m^2]
        Mc = 280      # Massa do carro [kg]
        C_alpha_d = 0.5  # Ângulo de Câmber Dianteiro [rad]
        C_alpha_t = 0.5  # Ângulo de Câmber Traseiro [rad]
        mu = 1.45        # Coeficiente de atrito

        # Slip angle e slip ratio constantes
        tire_Sa = np.deg2rad(9)  # Convertendo 9 graus para radianos
        front_tire_Ls = 0.10  # Slip ratio frontal
        rear_tire_Ls = 0.25   # Slip ratio traseiro

        # Criando instância do modelo
        params = (E, C_y, C_x, c1, c2)
        model = control(tire_Sa=tire_Sa, front_tire_Ls=front_tire_Ls, rear_tire_Ls=rear_tire_Ls,
                        tire_friction_coef=mu, params=params, mass=Mc, weight_distribution=0.5,
                        rear_Ca=C_alpha_t, front_Ca=C_alpha_d, Ld=Ld, Lt=Lt, W2=W2,
                        yaw_inertia=I_psi, roll_inertia=I_rho, pitch_inertia=I_beta, hcg=H_cg)
        
        # Vetor de tempo
        t = np.linspace(0, 10, 100)  # de 0 a 10 segundos

        # Calculando acelerações angulares
        yaw_ddot = []
        roll_ddot = []
        pitch_ddot = []
        for _ in t:
            yaw_val = model.yaw()
            roll_val = model.roll_moment()[0]
            pitch_val = model.pitch_moment()[0]
            yaw_ddot.append(yaw_val)
            roll_ddot.append(roll_val)
            pitch_ddot.append(pitch_val)

            model.load_transfer()

        yaw_ddot = np.array(yaw_ddot)
        roll_ddot = np.array(roll_ddot)
        pitch_ddot = np.array(pitch_ddot)

        # Integração para obter velocidades e deslocamentos (ângulos)
        yaw_dot, yaw_disp = control.integrate_motion(t, yaw_ddot)
        roll_dot, roll_disp = control.integrate_motion(t, roll_ddot)
        pitch_dot, pitch_disp = control.integrate_motion(t, pitch_ddot)

        # Plotando acelerações, velocidades e deslocamentos para yaw
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(t, yaw_ddot, label='Yaw Acceleration', color='r')
        plt.plot(t, yaw_dot, label='Yaw Velocity', linestyle='--', color='m')
        plt.plot(t, yaw_disp, label='Yaw Displacement', linestyle='-.', color='k')
        plt.xlabel('Tempo [s]')
        plt.ylabel('Yaw [rad, rad/s, rad/s^2]')
        plt.title('Yaw: Aceleração, Velocidade e Deslocamento')
        plt.legend()
        plt.grid()

        # Plotando acelerações, velocidades e deslocamentos para roll
        plt.subplot(3, 1, 2)
        plt.plot(t, roll_ddot, label='Roll Acceleration', color='g')
        plt.plot(t, roll_dot, label='Roll Velocity', linestyle='--', color='c')
        plt.plot(t, roll_disp, label='Roll Displacement', linestyle='-.', color='b')
        plt.xlabel('Tempo [s]')
        plt.ylabel('Roll [rad, rad/s, rad/s^2]')
        plt.title('Roll: Aceleração, Velocidade e Deslocamento')
        plt.legend()
        plt.grid()

        # Plotando acelerações, velocidades e deslocamentos para pitch
        plt.subplot(3, 1, 3)
        plt.plot(t, pitch_ddot, label='Pitch Acceleration', color='orange')
        plt.plot(t, pitch_dot, label='Pitch Velocity', linestyle='--', color='purple')
        plt.plot(t, pitch_disp, label='Pitch Displacement', linestyle='-.', color='brown')
        plt.xlabel('Tempo [s]')
        plt.ylabel('Pitch [rad, rad/s, rad/s^2]')
        plt.title('Pitch: Aceleração, Velocidade e Deslocamento')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

# Executa a função principal
control.model_input()

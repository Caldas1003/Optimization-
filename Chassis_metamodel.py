import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

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
        self.Mc = Mc      # Massa do carro [kg]
        self.Mst = Mst    # Massa suspensa traseira [kg]
        self.Msd = Msd    # Massa suspensa dianteira [kg]

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
        self.Kt = Kt      # Torcional [N/m]

        # Amortecedores da suspensão
        self.Cte = Cte    # Traseira esquerda [N·s/m]
        self.Ctd = Ctd    # Traseira direita [N·s/m]
        self.Cde = Cde    # Dianteira esquerda [N·s/m]
        self.Cdd = Cdd    # Dianteira direita [N·s/m]

        # Amortecedores estruturais do chassi
        self.Cphi = Cphi      # Torcional [N·s/m]
        self.Ctheta = Ctheta  # Flexional [N·s/m]

        # Inércias do chassi
        self.If_ = Iflex  # Inércia flexional [m⁴]
        self.It = Itorc   # Inércia torcional [m⁴]

    def spring(self, mass, k, x):

        F = k*x
        acceleration = F/mass

        return acceleration

    def damper(self, mass, c, xdot):

        F = c*xdot
        acceleration = F/mass

        return acceleration

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

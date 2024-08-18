import numpy as np
from scipy.integrate import solve_ivp

class MassaMolaAmortecedor:
    def __init__(self, m, k, c):
        self.m = m
        self.k = k
        self.c = c
        self.A = np.array([[0, 1], [-k / m, -c / m]])
        self.B = np.array([[0], [1 / m]])
        self.C = np.eye(2)
        self.D = np.zeros((2, 1))

    def sistema_dinamico(self, t, x, u, z):
        return self.A @ x + self.B @ u(t) + z(t)

    def simular(self, x0, t_span, u, z):
        sol = solve_ivp(self.sistema_dinamico, t_span, x0, args=(u, z), t_eval=np.linspace(t_span[0], t_span[1], 5000))
        return sol

import numpy as np

def entrada(t):
    return np.array([7000 + 2000 * np.sin(0.1 * t)])  # Força com uma oscilação de ±2000 N

# def entrada(t):
#     return np.array([7000 * np.exp(0.01 * t)])  # Força aumenta exponencialmente
# def entrada(t):
#     return np.array([7000 + 100 * t])  # Força aumenta 100 N por segundo
#
# def entrada(t):
#     amplitude = 3000 * np.sin(0.05 * t)  # Amplitude variável
#     frequency = 0.1  # Frequência em Hz
#     return np.array([7000 + amplitude * np.sin(2 * np.pi * frequency * t)])  # Força com variação senoidal
#
# def entrada(t):
#     return np.array([7000])  # Constante, ou ajuste conforme necessário

def trepidacao_pista(t):
    return 0.1 * np.sin(2 * np.pi * 0.1 * t)

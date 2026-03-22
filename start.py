import matplotlib.pyplot as plt
from datetime import datetime
from differencial_evolution import customDifferentialEvolution
import time
import numpy as np
import os
from evaluators import lapTime

def generations_to_plot(amount: int, step: int) -> list:
    return np.arange(1, amount + 1) * step


def save_results(
    best_params,
    best_fitness,
    standard_deviation,
    max_gen,
    pop_size,
    F,
    CR,
    elapsed_time,      # ⬅️ não usar "time" (conflito com módulo)
    time_stamp,
    folder_path
):
    """
    Salva os resultados da otimização em arquivo texto.
    """

    # -----------------------------
    # Organização dos dados
    # -----------------------------
    hours, minutes, seconds = elapsed_time

    parameters, speed_profile = best_params  # unpack correto

    filename = f"{time_stamp}_gen={max_gen}_pop={pop_size}.txt"
    file_path = os.path.join(folder_path, filename)

    # -----------------------------
    # Conteúdo do arquivo
    # -----------------------------
    content = []
    content.append("==== RESULTADOS DA OTIMIZAÇÃO ====\n")
    content.append(f"F = {F}\n")
    content.append(f"CR = {CR}\n")
    content.append(f"Gerações máximas = {max_gen}\n")
    content.append(f"Tamanho da população = {pop_size}\n\n")

    content.append("---- MELHORES PARÂMETROS ----\n")

    param_names = [
        "Distribuição de peso",
        "Kds (mola dianteira)",
        "Kts (mola traseira)",
        "Kpds (pneu dianteiro)",
        "Kpts (pneu traseiro)",
        "Kf (flexão do chassi)",
        "Kt (torção do chassi)",
        "Cds (amortecimento dianteiro)",
        "Cts (amortecimento traseiro)",
        "W (bitola)",
        "Lt (entre-eixos traseiro)",
        "Ld (entre-eixos dianteiro)"
    ]

    for name, value in zip(param_names, parameters):
        content.append(f"{name}: {value:.6e}\n")

    content.append("\n---- RESULTADOS ----\n")
    content.append(f"Menor tempo total encontrado: {best_fitness:.6f} s\n")
    content.append(f"Desvio padrão final: {standard_deviation:.6f}\n")
    content.append(
        f"Tempo total de execução: "
        f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}\n"
    )

    content.append("\n---- PERFIL DE VELOCIDADE ----\n")
    content.append(
        f"Velocidade mínima: {np.min(speed_profile):.3f} m/s\n"
        f"Velocidade máxima: {np.max(speed_profile):.3f} m/s\n"
        f"Velocidade média: {np.mean(speed_profile):.3f} m/s\n"
    )

    # -----------------------------
    # Escrita do arquivo
    # -----------------------------
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(content)

    print(f"Resultados salvos em: {file_path}")


def run():
    start_time = time.time()
    F = 0.6
    CR = 0.3
    max_gen = 10
    pop_size = 5
    track_start = 0
    track_end = 200
    gens_to_plot = generations_to_plot(20, 10)
    bounds = [
        [0.4, 0.6],         # distribution
        [1.5e4, 6.5e4],     # Rigidez da mola dianteira
        [1.5e4, 6.5e4],     # Kts
        [9.0e4, 1.5e5],     # Kpds
        [9.0e4, 1.5e5],     # Kpts
        [1e5, 5e7],         # Kf
        [0.8e5, 2.4e7],     # Kt
        [1.0e3, 5.0e3],     # Cds
        [1.0e3, 5.0e3],     # Cts
        [0.9, 1.1],         # W
        [1, 1.2],           # Lt
        [1.2, 1.6],         # Ld
        [0.278, 22.22],     # max speed
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d %H%M")
    folder = f"{timestamp} - speed profile gen"
    os.makedirs(folder, exist_ok=True)

    best_params, best_fitness, standard_deviation, data = customDifferentialEvolution(
        lapTime,
        np.array(bounds),
        max_generations=max_gen,
        F=F,
        CR=CR,
        pop_size=pop_size,
        track_start=0,
        track_end=200,
        gensToPlot=gens_to_plot,
        folder=folder
    )

    time_elapsed = time.time() - start_time
    hours, seconds_remain = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(seconds_remain, 60)

    save_results(
        best_params=best_params,
        best_fitness=best_fitness,
        standard_deviation=standard_deviation,
        max_gen=max_gen,
        pop_size=pop_size,
        F=F,
        CR=CR,
        elapsed_time=(hours, minutes, seconds),
        time_stamp=timestamp,
        folder_path=folder
    )
    print(f"Melhores parametros: {best_params[0]}")
    print(f"Speed profile: {best_params[1]}")
    print(f"Velocidade mínima: {best_params[1].min()}")
    print(f"Velocidade máxima: {best_params[1].max()}")
    print(f"Velocidade média: {best_params[1].mean()}")
    print(f"Menor tempo total encontrada: {best_fitness}")
    print(f"Desvio padrão: {standard_deviation}")
    print(f"Total time elapsed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    generations = [i + 1 for i in range(len(data[0]))]
    plt.figure(figsize=(18, 10))
    plt.suptitle(
        f"Tempo gasto: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\nDesvio padrão: {standard_deviation:.2f}\nMenor tempo: {best_fitness:.2f}\n",
        fontsize=16,
    )
    plt.subplots_adjust(wspace=0.8, hspace=1.2)

    plt.subplot(4, 1, 1)
    plt.plot(generations, data[0])
    plt.xlabel("Geração")
    plt.ylabel("Melhor tempo de volta")
    # plt.ylim(0 , 2500)
    # plt.xlim(0, max_gen)

    plt.subplot(4, 1, 2)
    plt.plot(generations, data[1])
    plt.xlabel("Geração")
    plt.ylabel("Desvio Padrão")
    plt.ylim(0, 10)
    # plt.xlim(0, max_gen)

    fileName = f"{timestamp} pop_size={pop_size} max_gen={max_gen} F={F} CR={CR}"

    plt.savefig(f"{folder}/charts.png")
    # TRACK.plotar_tracado_na_pista(f"somulação {timstamp}/{fileName} tracado.png", best_params[0], track_size)

if __name__ ==   '__main__':
    run()

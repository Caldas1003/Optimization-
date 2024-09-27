import numpy as np
from PistaComAtrito import TRACK

def build_checkpoints() -> list[list]:
    X_malha, Y_malha, Z_malha = TRACK.gerar_malha()
    X_left = X_malha[0][0]
    X_right = X_malha[0][1]
    Y_left = Y_malha[0][0]
    Y_right = Y_malha[0][1]
    Z_left = Z_malha[0][0]
    Z_right = Z_malha[0][1]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    gates = list()
    for i, value in enumerate(X_left):
        left_point = (X_left[i], Y_left[i], Z_left[i])
        right_point = (X_right[i], Y_right[i], Z_right[i])

        num_points = 100
        t = np.linspace(0, 1, num_points)
        waypoints = np.outer(1 - t, left_point) + np.outer(
            t, right_point
        )  # 100 points between the left and right points

        gates.append(waypoints)
        # ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2])

    # plt.savefig("gates.png")
    return gates

CHECKPOINTS = build_checkpoints()
import numpy as np

# A lógica por trás de ambas funções é a mesma, mas de maneira inversa
# Elas utilizam da recursividade para obter o maior valor possível para a aceleração ou o torque
# Para encontrar encontrar a maior aceleração a aumentamos gradualmente até que a condição seja quebrada
# Para encontrar encontrar o maior torque o diminuimos gradualmente até que a condição seja quebrada
# A condição é quebrada quando a força motriz é igual a força de atrito 
# Quando a condição não é verdade, a roda está girando em falso ou não estamos utilizando todo o atrito possível

def max_acceleration(suspension_stiffness, friction_coef, acceleration = 0):
    weigth_cg, weigth_front, weigth_rear, heigth_cg, cg_to_rear_length = 280, 120, 160, 0.2, 0.85

    driving_force = (weigth_cg*acceleration)
    mometum_from_driving_force = driving_force * heigth_cg
    downwards_force = mometum_from_driving_force / cg_to_rear_length
    rear_axle_vetical_displacement = downwards_force / suspension_stiffness
    # np.sqrt(1 - (x / Lr) ** 2) is the cosine of the angle due to the vertical displacement
    weigth_front_after_displacement = weigth_front * np.sqrt(1 - (rear_axle_vetical_displacement / cg_to_rear_length) ** 2)
    weigth_shifted = weigth_front - weigth_front_after_displacement
    total_force_on_rear_axle = weigth_rear + downwards_force + weigth_shifted
    friction_force = total_force_on_rear_axle * friction_coef
    maximum_acceleration = friction_force / weigth_cg

    if friction_force > driving_force:
        maximum_acceleration = max_acceleration(suspension_stiffness, friction_coef, maximum_acceleration)

    return maximum_acceleration

print(max_acceleration(10000, 0.8))

def max_torque(suspension_stiffness, friction_coef, wheel_torque):
    weigth_front, weigth_rear, heigth_cg, cg_to_rear_length, wheel_radius = 120, 160, 0.2, 0.85, 0.15

    driving_force = wheel_torque / wheel_radius
    mometum_from_driving_force = driving_force * heigth_cg
    downwards_force = mometum_from_driving_force / cg_to_rear_length
    rear_axle_vetical_displacement = downwards_force / suspension_stiffness
    # np.sqrt(1 - (x / Lr) ** 2) is the cosine of the angle due to the vertical displacement
    weigth_front_after_displacement = weigth_front * np.sqrt(1 - (rear_axle_vetical_displacement / cg_to_rear_length) ** 2)
    weigth_shifted = weigth_front - weigth_front_after_displacement
    total_force_on_rear_axle = weigth_rear + downwards_force + weigth_shifted
    friction_force = total_force_on_rear_axle * friction_coef
    maximum_torque = friction_force * wheel_radius

    if friction_force < driving_force:
        maximum_torque = max_torque(suspension_stiffness, friction_coef, maximum_torque)

    return maximum_torque

print(max_torque(100000, 0.8, 76))



def rear_axle_normal_force(rear_axle_vetical_displacement):
    '''
    Combine with the friction coeficiente given by the current track spot to calculate the maximum wheel torque possible
    '''
    weigth_front, weigth_rear, suspension_stiffness, cg_to_rear_length = 120, 160, 0.2, 0.85

    # np.sqrt(1 - (x / Lr) ** 2) is the cosine of the angle due to the vertical displacement
    weigth_front_after_displacement = weigth_front * np.sqrt(1 - (rear_axle_vetical_displacement / cg_to_rear_length) ** 2)
    weigth_shifted = weigth_front - weigth_front_after_displacement

    return weigth_rear + (rear_axle_vetical_displacement * suspension_stiffness) + weigth_shifted
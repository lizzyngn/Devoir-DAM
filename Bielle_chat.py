from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import numpy as np

# Paramètres du moteur
D = 73e-3
C = 74.2e-3
L = 166.4e-3
mpiston = 0.3
mbielle = 0.6
tau = 11
Mair_carb = 14.5
R = C / 2
V_c = np.pi * (D**2) * R / 2
beta = L / R
gamma = 1.3

# Apport de chaleur
def Q(s):
    PCI = 44000000
    R_1 = 8.314
    M_air = 28.96
    p_1 = 100000 * s
    T_1 = 303.15
    masse_vol_air = (p_1 * M_air) / (R_1 * T_1)
    return (V_c * masse_vol_air * PCI) / Mair_carb

def Q_output(theta, thetaC, deltaThetaC, s):
    return np.where((theta > thetaC) & (theta < thetaC + deltaThetaC),
                    (Q(s) / 2) * (1 - np.cos(np.pi * ((theta - thetaC) / deltaThetaC))), 0)

def dQsurdtheta(theta, thetaC, deltaThetaC, s):
    return np.where((theta > thetaC) & (theta < thetaC + deltaThetaC),
                    (Q(s) / 2) * np.sin((np.pi * (theta - thetaC)) / deltaThetaC) * (np.pi / deltaThetaC), 0)

# Volume du cylindre
def V_output(theta):
    theta_rad = np.radians(theta)
    return (V_c / 2) * (1 - np.cos(theta_rad) + beta - np.sqrt(beta**2 - np.sin(theta_rad)**2)) + (V_c / (tau - 1))

def dVolumesurdtheta(theta):
    theta_rad = np.radians(theta)
    return (V_c / 2) * (np.sin(theta_rad) + (np.sin(theta_rad) * np.cos(theta_rad)) / (np.sqrt(beta**2 - np.sin(theta_rad)**2)))

# Pression
def dpsurdtheta(theta, p, thetaC, deltaThetaC, s):
    return -gamma * (p / V_output(theta)) * dVolumesurdtheta(theta) + (gamma - 1) * (1 / V_output(theta)) * Q_output(theta, thetaC, deltaThetaC, s)

def pression(theta, thetaC, deltaThetaC, s):
    p_1 = 100000 * s
    solution = solve_ivp(dpsurdtheta, (theta[0], theta[-1]), [p_1], t_eval=theta, method='RK45',
                         args=(thetaC, deltaThetaC, s))
    return solution.y[0]

# Forces appliquées sur la bielle
def F_pied_output(rpm, s, theta, thetaC, deltaThetaC):
    omega = (rpm * 2 * np.pi) / 60
    return ((np.pi * (D**2)) / 4) * pression(theta, thetaC, deltaThetaC, s) - mpiston * R * omega**2 * np.cos(np.radians(theta))

def F_tete_output(rpm, s, theta, thetaC, deltaThetaC):
    omega = (rpm * 2 * np.pi) / 60
    return -((np.pi * (D**2)) / 4) * pression(theta, thetaC, deltaThetaC, s) + (mpiston + mbielle) * R * omega**2 * np.cos(np.radians(theta))

def F_crit(rpm, s, theta, thetaC, deltaThetaC):
    return np.max((F_pied_output(rpm, s, theta, thetaC, deltaThetaC) + F_tete_output(rpm, s, theta, thetaC, deltaThetaC)) / 2)

# Épaisseur critique de la bielle
def equation_de_rankine(t, F_critique):
    E = 200e9
    sigma_C = 450e6
    K_x, K_y = 1, 0.5
    t_x, t_y = t if isinstance(t, (list, tuple, np.ndarray)) else (t, t)
    I_x = (419/12) * (t_x**4)
    I_y = (131/12) * (t_y**4)
    A_x = 11 * (t_x**2)
    A_y = 11 * (t_y**2)
    F_euler_x = ((np.pi**2) * E * I_x) / ((K_x * L)**2)
    F_euler_y = ((np.pi**2) * E * I_y) / ((K_y * L)**2)
    eq_x = (1 / F_critique) - (1 / F_euler_x) - (1 / (A_x * sigma_C))
    eq_y = (1 / F_critique) - (1 / F_euler_y) - (1 / (A_y * sigma_C))
    return np.array([eq_x, eq_y])

def t(rpm, s, theta, thetaC, deltaThetaC):
    F_critique = F_crit(rpm, s, theta, thetaC, deltaThetaC)
    t_init = [0.005, 0.005]
    solution = fsolve(equation_de_rankine, t_init, args=(F_critique,))
    return max(solution)

def myfunc(rpm, s, theta, thetaC, deltaThetaC):
    V = V_output(theta)
    Q = Q_output(theta, thetaC, deltaThetaC, s)
    p = pression(theta, thetaC, deltaThetaC, s)
    F_pied = F_pied_output(rpm, s, theta, thetaC, deltaThetaC)
    F_tete = F_tete_output(rpm, s, theta, thetaC, deltaThetaC)
    t_res = t(rpm, s, theta, thetaC, deltaThetaC)
    return V, Q, F_pied, F_tete, p, t_res

# Test
theta_test = np.radians(np.linspace(-180, 180, 360))
V, Q, F_pied, F_tete, p, epaisseur = myfunc(3000, 1.2, theta_test, 30, 40)
print("Volume du cylindre:", V[:5])
print("Apport de chaleur:", Q[:5])
print("Force au pied de bielle:", F_pied[:5])
print("Force à la tête de bielle:", F_tete[:5])
print("Pression dans le cylindre:", p[:5])
print("Épaisseur critique de la bielle:", epaisseur)

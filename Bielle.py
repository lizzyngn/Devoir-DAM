from scipy.integrate import solve_ivp
import numpy as np

# Grandeurs géométriques du moteur
D = None #@valeur alesage@ #[m]
C = None #@valeur course@ #[m]
L = None #@valeur longueur bielle@ #[m]
mpiston = None #@valeur masse piston@ #[kg]
mbielle = None #@valeur masse bielle@ #[kg]
tau = None #@valeur taux compression@ #[-]
Mair_carb = None #@valeur melange air carburant, 14.5[kg_air/kg_fuel] pour un moteur essence ou 26 [kg_air/kg_fuel] pour un moteur diesel  @ #[kg_air/kg_fuel]
R = C/2 #longueur de la manivelle qui vaut la moitié de la course du piston dans le cylindre
V_c = np.pi*(D*D)*R/2 #volume balayé par le piston lors d'une course complète
beta = L/R
gamma = 1.3

def Q(s):
    PCI = 44000000 #valeur du pouvoir calorifique inférieur de l'essence #[J/kg]
    R_1 = 8.314 #constante des gaz parfaits [J/(mol*K)]
    M_air = 28,96  #masse molaire de l'air [g/mol]
    p_1 = 10000 * s #pression de l'air [Pa]
    T_1 = #jsp comment le trouver juste qu'il faut utiliser gamma = 1.3 #temperature [K]
    masse_vol_air = (p_1 * M_air)/(R_1*T_1)
    return (V_c * masse_vol_air * PCI)/Mair_carb

def Q_output(theta, thetaC, deltaThetaC, s):
    return np.where((theta> thetaC) & (theta < thetaC+deltaThetaC), (Q(s)/2) *(1- np.cos(np.pi*((theta - thetaC)/deltaThetaC))), 0)

def V_output(theta):
    return (V_c/2)*(1-np.cos(theta) + beta - np.sqrt(beta*beta - np.sin(theta)*np.sin(theta))) +(V_c/(tau-1))

def dVolumesurdtheta(theta):
    return (V_c/2)*(np.sin(theta)+(np.sin(theta)*np.cos(theta))/(np.sqrt(beta*beta - np.sin(theta) * np.sin(theta))))

def dQsurdtheta(theta, thetaC, deltaThetaC, s):
    return np.where((theta> thetaC) & (theta < thetaC+deltaThetaC), (Q(s)/2) * np.sin((np.pi *(theta-thetaC))/deltaThetaC) *(np.pi/deltaThetaC), 0)

def dpsurdtheta(theta, thetaC, deltaThetaC, s, p):
    return -gamma * p/V_output(theta)*dVolumesurdtheta(theta) +(gamma - 1) * 1/V_output(theta)*Q_output(theta, thetaC, deltaThetaC, s)

def pression(theta, thetaC, deltaThetaC, s):

    p_1 = 10000 * s
    theta_span = (theta[0], theta[-1])

    # Résolution
    solution = solve_ivp(dpsurdtheta, theta_span, p_1, t_eval=theta, method='RK45')

    # Résultats
    p_solution = solution.y[0]
    return p_solution
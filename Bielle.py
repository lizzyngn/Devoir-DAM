from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import numpy as np

# Grandeurs géométriques du moteur basée sur le moteur Suzuki Swift II essence
D = 73e-3 #e@valeur alesage@ #[m]
C = 74.2e-3 #@valeur course@ #[m]
L = 166.4e-3 #@valeur longueur bielle@ #[m]
mpiston = None #@valeur masse piston@ #[kg]
mbielle = None #@valeur masse bielle@ #[kg]
tau = 11  #@valeur taux compression@ #[-] Vmax/Vmin
Mair_carb = 14.5 #@valeur melange air carburant, 14.5[kg_air/kg_fuel] pour un moteur essence ou 26 [kg_air/kg_fuel] pour un moteur diesel  @ #[kg_air/kg_fuel]

R = C/2 # Longueur de la manivelle qui vaut la moitié de la course du piston dans le cylindre
V_c = np.pi*(D*D)*R/2 # Volume balayé par le piston lors d'une course complète
beta = L/R # Rapport longueur bielle/manivelle
gamma = 1.3 # Coefficient isentropique

# Evolution de l'apport de chaleur
def Q(s):
    PCI = 44000000 #valeur du pouvoir calorifique inférieur de l'essence #[J/kg]
    R_1 = 8.314 #constante des gaz parfaits [J/(mol*K)]
    M_air = 28.96  #masse molaire de l'air [g/mol]
    p_1 = 100000 * s #pression d'admission, avec p_atm = 100000, [Pa]
    T_1 = 303.15
    masse_vol_air = (p_1 * M_air)/(R_1*T_1)
    return (V_c * masse_vol_air * PCI)/Mair_carb

def Q_output(theta, thetaC, deltaThetaC, s):
    return np.where((theta> thetaC) & (theta < thetaC+deltaThetaC), (Q(s)/2) *(1- np.cos(np.pi*((theta - thetaC)/deltaThetaC))), 0)

def dQsurdtheta(theta, thetaC, deltaThetaC, s):
    return np.where((theta> thetaC) & (theta < thetaC+deltaThetaC), (Q(s)/2) * np.sin((np.pi *(theta-thetaC))/deltaThetaC) *(np.pi/deltaThetaC), 0)

# Evolution du volume du cylindre
def V_output(theta):
    return (V_c/2)*( 1 - np.cos(theta) + beta - np.sqrt(beta*beta - np.sin(theta)*np.sin(theta))) +(V_c/(tau-1))

def dVolumesurdtheta(theta):
    return (V_c/2)*(np.sin(theta)+(np.sin(theta)*np.cos(theta))/(np.sqrt(beta*beta - np.sin(theta) * np.sin(theta))))

# Evolution de la pression dans le cylindre
def dpsurdtheta(theta, thetaC, deltaThetaC, s, p):
    return -gamma * (p/V_output(theta))*dVolumesurdtheta(theta) +(gamma - 1) * (1/V_output(theta))*Q_output(theta, thetaC, deltaThetaC, s)

def pression(theta, thetaC, deltaThetaC, s): #j'ai changé cette partie car il y avait une erreur qui s'affichait
    #car il manquait une partie de la ligne
    p_1 = 100000 * s  # Pression initiale [Pa]
    solution = solve_ivp(dpsurdtheta, (theta[0], theta[-1]), [p_1], t_eval=theta, method='RK45',
                         args=(thetaC, deltaThetaC, s))
    return solution.y[0]

# Evolution des forces s'appliquant sur la bielle
# Vitesse normale : 3000 rpm, vitesse élevée : 5000 rpm
def F_pied_output(rpm, s, theta, thetaC, deltaThetaC):
    omega = (rpm * 2 * np.pi) / 60  # Conversion rpm en rad/s
    return ((np.pi * (D**2)) / 4) * pression(theta, thetaC, deltaThetaC, s) - mpiston * R * omega*omega * np.cos(theta)

def F_tete_output(rpm, s, theta, thetaC, deltaThetaC):
    omega = (rpm * 2 * np.pi) / 60  # Conversion rpm en rad/s
    return -((np.pi * (D**2)) / 4) * pression(theta, thetaC, deltaThetaC, s) + (mpiston + mbielle) * R * omega*omega * np.cos(theta)

# Epaisseur critique de la bielle (flambage)
def F_crit(rpm, s, theta, thetaC, deltaThetaC) :
    return np.max((F_pied_output(rpm, s, theta, thetaC, deltaThetaC) + F_tete_output(rpm, s, theta, thetaC, deltaThetaC)) / 2)

def equation_de_rankine(t, F_critique):
    E = 200e9  # Module d'élasticité [Pa]
    sigma_C = 450e6  # Résistance à la compression [Pa]
    K_x, K_y = 1, 0.5  # Facteurs de correction selon la direction
    I_x = (419/12)*(t**4)  # Moment d’inertie en x
    I_y = (131/12)*(t**4)  # Moment d’inertie en y
    A = 11 * (t ** 2) # Aire de la bielle
    F_euler_x = ((np.pi * np.pi) * E * I_x)/((K_x*L)**2) # Force d'Euler selon x
    F_euler_y = ((np.pi * np.pi) * E * I_y)/((K_y*L)**2) # Force d'Euler selon y

    # Equation de Rankine dans les deux directions
    eq_x = (1 / F_critique) - (1 / F_euler_x) - (1 / (A * sigma_C))
    eq_y = (1 / F_critique) - (1 / F_euler_y) - (1 / (A * sigma_C))

    return [eq_x, eq_y]

def t(rpm, s, theta, thetaC, deltaThetaC):
    F_critique = F_crit(theta, rpm, thetaC, deltaThetaC, s)
    
    # Résolution de l'équation de Rankine
    t_init = 0.0  # 0.005?
    solution = fsolve(equation_de_rankine, [t_init, t_init], args=(F_critique,))
    
    t_max = max(solution)
    
    return t_max

def myfunc(rpm, s, theta, thetaC, deltaThetaC):
    
    # Calcul des résultats
    V = V_output(theta)
    Q = Q_output(theta, thetaC, deltaThetaC, s)
    p = pression(theta, thetaC, deltaThetaC, s)
    F_pied = F_pied_output(rpm, s, theta, thetaC, deltaThetaC)
    F_tete = F_tete_output(rpm, s, theta, thetaC, deltaThetaC)
    t_res = t(rpm, s, theta, thetaC, deltaThetaC)
    
    return V, Q, F_pied, F_tete, p, t_res

# Test (chat)
theta_test = np.linspace(-180, 180, 360)
V, Q, F_pied, F_tete, p, epaisseur = myfunc(3000, 1.2, theta_test, 30, 40)
print("Volume du cylindre:", V[:5])
print("Apport de chaleur:", Q[:5])
print("Force au pied de bielle:", F_pied[:5])
print("Force à la tête de bielle:", F_tete[:5])
print("Pression dans le cylindre:", p[:5])
print("Épaisseur critique de la bielle:", epaisseur)

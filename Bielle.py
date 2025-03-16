from scipy.integrate import solve_ivp
import numpy as np

# Grandeurs géométriques du moteur
D = None #@valeur alesage@ #[m]
C = None #@valeur course@ #[m]
L = None #@valeur longueur bielle@ #[m]
mpiston = None #@valeur masse piston@ #[kg]
mbielle = None #@valeur masse bielle@ #[kg]
Vmax=  None#à déterminer
Vmin= None #à déterminer
tau = Vmax/Vmin  #@valeur taux compression@ #[-]
Mair_carb = None #@valeur melange air carburant, 14.5[kg_air/kg_fuel] pour un moteur essence ou 26 [kg_air/kg_fuel] pour un moteur diesel  @ #[kg_air/kg_fuel]

R = C/2 #longueur de la manivelle qui vaut la moitié de la course du piston dans le cylindre
V_c = np.pi*(D*D)*R/2 #volume balayé par le piston lors d'une course complète
beta = L/R # Rapport longueur bielle/manivelle
gamma = 1.3

def Q(s):
    PCI = 44000000 #valeur du pouvoir calorifique inférieur de l'essence #[J/kg]
    R_1 = 8.314 #constante des gaz parfaits [J/(mol*K)]
    M_air = 28,96  #masse molaire de l'air [g/mol]
    p_1 = 10000 * s #pression de l'air [Pa]
    T_1 = None #à déterminer peut être avec la formule _admission * (s) ** ((gamma - 1) / gamma)
#jsp comment le trouver juste qu'il faut utiliser gamma = 1.3 #temperature [K]
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

def pression(theta, thetaC, deltaThetaC, s): #j'ai changé cette partie car il y avait une erreur qui s'affichait
    #car il manquait une partie de la ligne
    p_1 = 100000 * s  # Pression initiale [Pa]
    solution = solve_ivp(dpsurdtheta, (theta[0], theta[-1]), [p_1], t_eval=theta, method='RK45',
                         args=(thetaC, deltaThetaC, s))
    return solution.y[0]

def myfunc(rpm, s, theta, thetaC, deltaThetaC):
    
    # Calcul des résultats
    V_output_res = V_output(theta)
    Q_output_res = Q_output(theta, thetaC, deltaThetaC, s)
    p_output = pression(theta, thetaC, deltaThetaC, s)
    
    
    # Forces sur la bielle
    omega = (rpm * 2 * np.pi) / 60  # Conversion rpm en rad/s
    F_pied_output = (np.pi * (D**2) / 4) * p_output - mpiston * R * omega**2 * np.cos(theta)
    F_tete_output = -(np.pi * (D**2) / 4) * p_output + (mpiston + mbielle) * R * omega**2 * np.cos(theta)
    
    # Épaisseur critique de la bielle (flambage)
    F_crit = np.max((F_pied_output + F_tete_output) / 2)
    E = 200e9  # Module d'élasticité [Pa]
    sigma_C = 450e6  # Résistance à la compression [Pa]
    K_x, K_y = 1, 0.5  # Facteurs de correction selon la direction
    I_x = (419/12)  # Facteur du moment d’inertie en x
    I_y = (131/12)  # Facteur du moment d’inertie en y
    t_x = ((np.pi**2 * E * I_x) / (K_x * L)**2 + 1 / (np.pi * sigma_C))**0.25
    t_y = ((np.pi**2 * E * I_y) / (K_y * L)**2 + 1 / (np.pi * sigma_C))**0.25 
    t = max(t_x, t_y)  #ici j'ai un doute que ce soit le max parce que quand je lui avit parlé du K
    #il m'avait dit de prendre 1 donc il y a moyen que ce soit tjs t_x
    
    return V_output_res, Q_output_res, F_pied_output, F_tete_output, p_output, t



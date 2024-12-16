import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import os  # For handling directories
from fokker_planck_simulator import FokkerPlanckSimulator
from Symbolic_Time_Deriv_Solver import SymbolicSolver
import sympy as sp

def solve_time_deriv_sym(symsolver, PHYS_model, prob_func_sym):  
    # Call the Fokker-Planck solver to get the time derivative functions
    TD_consts = symsolver.one_mode_fokker_planck(PHYS_model, prob_func_sym, x_sym, y_sym, 
                                                 a, b, c, d, e, f)

    # Unpack the time derivatives
    TD = [
        TD_consts[i] for i in range(6)
    ]
    
    # Create symbolic variables for the system_time_evolution
    symbols = (a, b, c, d, e, f, gamma_sym, nu_sym, n_th_sym)

    # Create lambdified functions for all derivatives using a loop
    time_deriv_funcs = [
        sp.lambdify(symbols, TD[i], 'numpy') for i in range(6)
    ]
    
    return tuple(time_deriv_funcs)

def system_time_evolution(t, y, phys_parameter, time_deriv_funcs):
    # Unpack physical parameters
    gamma, nu, n_th= phys_parameter

    # Unpack state variables
    a0, b0, c0, d0, e0, f0= y

    # Unpack the derivative functions
    derivatives = [func(a0, b0, c0, d0, e0, f0, gamma, nu, n_th)
                   for func in time_deriv_funcs]

    return np.array(derivatives)


# Initialize simulation parameters
#   Physical Parameter
gamma = 1.0
nu = 3
n_th = 0

eps2 = 1
x0 = 2
y0 = 2

phys_parameter = gamma, nu, n_th

Dim = 2
a0 = - (x0 * x0 + y0 * y0)/eps2  - Dim * np.log( np.sqrt(np.pi * eps2)) # Initial condition for a
b0 = 2 * x0 /eps2  # Initial condition for b
c0 = 2 * y0 /eps2  # Initial condition for c
d0 = - 1/eps2  # Initial condition for d
e0 = - 1/eps2  # Initial condition for e
f0 = 0
init_cond = np.array([a0, b0, c0, d0, e0, f0])

#   Time parameter
t_start = 0
t_end = 10
dt= 0.01
simulation_time_setting = t_start, t_end, dt

#   Map Parameter
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
simulation_grid_setting = x, y

#   Output directory
output_dir = "Q_rep_PhotonDissipation"

# Symbolic formula solver settings
#   Choice of Probability Representation
representation = 'Q'
SymSolver = SymbolicSolver()
gamma_sym, nu_sym, n_th_sym = sp.symbols('gamma_sym nu_sym n_th_sym')

def Photon_QREP_FKEQ(x,y,Px,Py,Pxx,Pyy,Pxy):
    TD1 = gamma_sym + (gamma_sym / 2 * x - nu_sym * y) * Px + (gamma_sym / 2 * y + nu_sym * x) * Py
    TD2 = gamma_sym * (n_th_sym+1) / 4 * (Pxx + Pyy)
    return TD1 + TD2

x_sym, y_sym, a, b, c, d, e, f = sp.symbols('x_sym y_sym a b c d e f')

# Use exponential function to simulate the solution of Fokker-Planck eq.
ProbFunc = sp.exp(a + b*x_sym + c*y_sym + d*x_sym**2 + e*y_sym**2 + f*x_sym*y_sym)

# Probability representation mapping mode
probdensmap_mode = '2D'

# Solve the symbolic formula of the time derivatives of the evolving parameters
time_deriv_funcs = solve_time_deriv_sym(SymSolver, Photon_QREP_FKEQ, ProbFunc)

# Instantiate and run the simulation
simulator = FokkerPlanckSimulator(representation, simulation_time_setting, simulation_grid_setting, phys_parameter, init_cond, output_dir, probdensmap_mode, system_time_evolution, time_deriv_funcs)
simulator.run_simulation(pure_parameter = False)
simulator.electric_field_evolution()
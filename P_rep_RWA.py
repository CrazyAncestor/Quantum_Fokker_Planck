import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import os  # For handling directories
from fokker_planck_simulator import FokkerPlanckSimulator
from Symbolic_Time_Deriv_Solver import SymbolicSolver
import sympy as sp

def generate_time_deriv_funcs(symsolver, PHYS_model, prob_func_sym):  
    # Call the Fokker-Planck solver to get the time derivative functions
    TD_consts = symsolver.two_mode_fokker_planck(PHYS_model, prob_func_sym, x_sym, y_sym, u_sym, v_sym, 
                                                 a, b, c, d, e, f, h, k, l, o, p, q, r, s, w)

    # Unpack the time derivatives
    TD = [
        TD_consts[i] for i in range(15)
    ]
    
    # Create symbolic variables for the system
    symbols = (a, b, c, d, e, f, h, k, l, o, p, q, r, s, w, gamma_sym, eta_sym, nu_sym, omega_sym, 
               n_th_sym, m_th_sym, g_sym)

    # Create lambdified functions for all derivatives using a loop
    time_deriv_funcs = [
        sp.lambdify(symbols, TD[i], 'numpy') for i in range(15)
    ]
    
    return tuple(time_deriv_funcs)

def system(t, y, phys_parameter, time_deriv_funcs):
    # Unpack physical parameters
    gamma, eta, nu, omega, n_th, m_th, g= phys_parameter

    # Unpack state variables
    a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0 = y

    # Unpack the derivative functions
    derivatives = [func(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)
                   for func in time_deriv_funcs]

    return np.array(derivatives)

# Initialize simulation parameters
#   Physical Parameter
gamma = 0.5
eta = 0.5
nu = 2 * np.pi
omega = 2 * np.pi
g =  2 * np.pi * 0.1
n_th = 0.01
m_th = 0.01

eps2 = 1
x0 = 2
y0 = 2

phys_parameter = gamma, eta, nu, omega, n_th, m_th, g

# Initial condition
Dim = 4
a0 = - (x0 * x0 + y0 * y0)/eps2 - Dim * np.log( np.sqrt(np.pi * eps2))

b0 = 2 * x0 /eps2
c0 = 2 * y0 /eps2
d0 = 0 * x0 /eps2
e0 = 0 * y0 /eps2

f0 = - 1/eps2
h0 = - 1/eps2
k0 = - 1/eps2
l0 = - 1/eps2

o0 = 0
p0 = 0
q0 = 0
r0 = 0
s0 = 0
w0 = 0

init_cond = np.array([a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0])

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
output_dir = "P_rep_RWA"

# Symbolic formula solver settings
#   Choice of Probability Representation
representation = 'P'
SymSolver = SymbolicSolver()
gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym = sp.symbols('gamma_sym eta_sym nu_sym omega_sym n_th_sym m_th_sym g_sym')

def RWA_PREP(x,y,u,v,Px,Py,Pu,Pv,Pxx,Pyy,Puu,Pvv,Pxy,Puv,Pxu,Pyu,Pxv,Pyv):
    TD1 = (gamma_sym + eta_sym) + (gamma_sym/2 * x - nu_sym * y - g_sym * v) * Px + (gamma_sym/2 * y + (nu_sym) * x + g_sym * u) * Py
    TD2 = (eta_sym/2 * u - (omega_sym) * v - g_sym *y) * Pu + (eta_sym/2 * v + (omega_sym) * u + g_sym * x) * Pv
    TD3 = gamma_sym * (n_th_sym) / 4 * (Pxx + Pyy) + eta_sym * (m_th_sym) / 4 * (Puu + Pvv)
    return TD1 + TD2 + TD3  

x_sym, y_sym, u_sym, v_sym, a, b, c, d, e, f, h, k, l, o, p, q, r, s, w = sp.symbols('x_sym y_sym u_sym v_sym a b c d e f h k l o p q r s w')

# Use exponential function to simulate the solution of Fokker-Planck eq.
ProbFunc = sp.exp(a + b*x_sym + c*y_sym + d*u_sym + e*v_sym + f*x_sym**2 + h*y_sym**2 + k*u_sym**2 + l*v_sym**2 + o*x_sym*y_sym + p*u_sym*v_sym +q*x_sym*u_sym + r*y_sym*u_sym + s*x_sym*v_sym +w*y_sym*v_sym)

# Probability representation mapping mode
probdensmap_mode = '4D'

# Solve the symbolic formula of the time derivatives of the evolving parameters
time_deriv_funcs = generate_time_deriv_funcs(SymSolver, RWA_PREP, ProbFunc)

# Instantiate and run the simulation
simulator = FokkerPlanckSimulator(representation, simulation_time_setting, simulation_grid_setting, phys_parameter, init_cond, output_dir, probdensmap_mode, system, time_deriv_funcs)
simulator.run_simulation(pure_parameter = False)
simulator.electric_field_evolution()
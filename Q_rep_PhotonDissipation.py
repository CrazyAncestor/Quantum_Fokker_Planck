import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import os  # For handling directories
from fokker_planck_simulator import FokkerPlanckSimulator
from Symbolic_Time_Deriv_Solver import SymbolicSolver
import sympy as sp

gamma_sym, nu_sym, n_th_sym = sp.symbols('gamma_sym nu_sym n_th_sym')

# Use exponential function to simulate the solution of Fokker-Planck eq.
def PT_model_QREP(x,y,Px,Py,Pxx,Pyy,Pxy):
    PT1 = gamma_sym + (gamma_sym / 2 * x - nu_sym * y) * Px + (gamma_sym / 2 * y + nu_sym * x) * Py
    PT2 = gamma_sym * (n_th_sym + 1) / 4 * (Pxx + Pyy)
    return PT1 + PT2

x_sym, y_sym, a, b, c, d, e, f = sp.symbols('x_sym y_sym a b c d e f')

def generate_time_deriv_funcs(symsolver, PT_model):  
      
    PT_const, PTx, PTy, PTxx, PTyy, PTxy = symsolver.solve_fkpk_symbol(PT_model, x_sym, y_sym, a, b, c, d, e, f)

    a_prime = sp.lambdify((a,b,c,d,e,f, gamma_sym, nu_sym, n_th_sym), PT_const, 'numpy')
    b_prime = sp.lambdify((a,b,c,d,e,f, gamma_sym, nu_sym, n_th_sym), PTx, 'numpy')
    c_prime = sp.lambdify((a,b,c,d,e,f, gamma_sym, nu_sym, n_th_sym), PTy, 'numpy')
    d_prime = sp.lambdify((a,b,c,d,e,f, gamma_sym, nu_sym, n_th_sym), PTxx, 'numpy')
    e_prime = sp.lambdify((a,b,c,d,e,f, gamma_sym, nu_sym, n_th_sym), PTyy, 'numpy')
    f_prime = sp.lambdify((a,b,c,d,e,f, gamma_sym, nu_sym, n_th_sym), PTxy, 'numpy')
    
    time_deriv_funcs = a_prime, b_prime, c_prime, d_prime, e_prime, f_prime

    return time_deriv_funcs

# RK4 Solver
def rk4_step(t, y, dt, phys_parameter, time_deriv_funcs):
    # Get the k1, k2, k3, k4 slopes
    k1 = dt* system(t, y, phys_parameter, time_deriv_funcs)
    k2 = dt* system(t + dt/2, y + k1/2, phys_parameter, time_deriv_funcs)
    k3 = dt* system(t + dt/2, y + k2/2, phys_parameter, time_deriv_funcs)
    k4 = dt* system(t + dt, y + k3, phys_parameter, time_deriv_funcs)
    
    # Update the solution using the weighted average
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# Define the system of differential equations
def system(t, y, phys_parameter, time_deriv_funcs):
    gamma, nu, n_th =  phys_parameter
    a0, b0, c0, d0, e0, f0 = y
    a_prime, b_prime, c_prime, d_prime, e_prime, f_prime = time_deriv_funcs
    
    a_prime0 = a_prime(a0,b0,c0,d0,e0,f0,gamma,nu,n_th)
    b_prime0 = b_prime(a0,b0,c0,d0,e0,f0,gamma,nu,n_th)
    c_prime0 = c_prime(a0,b0,c0,d0,e0,f0,gamma,nu,n_th)
    d_prime0 = d_prime(a0,b0,c0,d0,e0,f0,gamma,nu,n_th)
    e_prime0 = e_prime(a0,b0,c0,d0,e0,f0,gamma,nu,n_th)
    f_prime0 = f_prime(a0,b0,c0,d0,e0,f0,gamma,nu,n_th)
    
    return np.array([a_prime0, b_prime0, c_prime0, d_prime0, e_prime0, f_prime0])

# Plotting complex representation function with parameters
def ProbDensMap(x, y, solution):
    X, Y = np.meshgrid(x, y)
    a, b, c, d, e, f = solution
    ProbDens = np.exp(a + b * X + c * Y + d * X**2 + e * Y**2 + f * X* Y)
    return ProbDens

def intensity(a,astar):
    return a*astar

# Initialize simulation parameters
#   Physical Parameter
gamma = 1.0
nu = 3
n_th = 3

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

#   Map Parameter
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

output_dir = "Q_rep_PhotonDissipation"

# Instantiate and run the simulation
SymSolver = SymbolicSolver()
time_deriv_funcs = generate_time_deriv_funcs(SymSolver, PT_model_QREP)
simulator = FokkerPlanckSimulator(t_start, t_end, dt, x, y, phys_parameter, init_cond, output_dir, ProbDensMap, rk4_step, time_deriv_funcs)
simulator.run_simulation(pure_parameter = True)
simulator.electric_field_evolution(representation='Q')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import os  # For handling directories
from fokker_planck_simulator import FokkerPlanckSimulator
import sympy as sp

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
    gamma0, nu0, n_th0, eps2, x0, y0 =  phys_parameter
    a0, b0, c0, d0, e0, f0 = y
    a_prime, b_prime, c_prime, d_prime, e_prime, f_prime = time_deriv_funcs
    
    a_prime0 = a_prime(a0,b0,c0,d0,e0,f0,gamma0,nu0,n_th0)
    b_prime0 = b_prime(a0,b0,c0,d0,e0,f0,gamma0,nu0,n_th0)
    c_prime0 = c_prime(a0,b0,c0,d0,e0,f0,gamma0,nu0,n_th0)
    d_prime0 = d_prime(a0,b0,c0,d0,e0,f0,gamma0,nu0,n_th0)
    e_prime0 = e_prime(a0,b0,c0,d0,e0,f0,gamma0,nu0,n_th0)
    f_prime0 = f_prime(a0,b0,c0,d0,e0,f0,gamma0,nu0,n_th0)
    
    return np.array([a_prime0, b_prime0, c_prime0, d_prime0, e_prime0, f_prime0])

# Plotting complex representation function with parameters
def ProbDensMap(x, y, solution):
    X, Y = np.meshgrid(x, y)
    a, b, c, d, e, f = solution
    ProbDens = np.exp(a + b * X + c * Y + d * X**2 + e * Y**2 + f * X* Y)
    return ProbDens

"""# Analytical solution function
def AnalyticalSol(x, y, t, phys_parameter):
    X, Y = np.meshgrid(x, y)
    gamma, nu, n_th, eps2, x0, y0 =  phys_parameter
    t0 = -np.log(1 - eps2/n_th) / gamma
    Dt = n_th * (1 - np.exp(- gamma * (t + t0)))

    xbar = (x0 * np.cos(nu*t) + y0 * np.sin(nu*t)) * np.exp(-gamma / 2 * t)
    ybar = (y0 * np.cos(nu*t) - x0 * np.sin(nu*t)) * np.exp(-gamma / 2 * t)

    ProbDens = np.exp(-((X-xbar)**2 + (Y-ybar)**2)/Dt) / np.pi / Dt
    return ProbDens
"""
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

phys_parameter = gamma, nu, n_th, eps2, x0, y0

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


simulator = FokkerPlanckSimulator(t_start, t_end, dt, x, y, phys_parameter, init_cond, output_dir, ProbDensMap, rk4_step)
simulator.run_simulation(pure_parameter = True)
"""simulator.electric_field_evolution(representation='Q')"""
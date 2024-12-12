import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import os  # For handling directories
from fokker_planck_simulator import FokkerPlanckSimulator

# RK4 Solver
def rk4_step(t, y, dt, phys_parameter):
    # Get the k1, k2, k3, k4 slopes
    k1 = dt* system(t, y, phys_parameter)
    k2 = dt* system(t + dt/2, y + k1/2, phys_parameter)
    k3 = dt* system(t + dt/2, y + k2/2, phys_parameter)
    k4 = dt* system(t + dt, y + k3, phys_parameter)
    
    # Update the solution using the weighted average
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6


# Define the system of differential equations
def system(t, y, phys_parameter):
    gamma, D, n_th =  phys_parameter
    a, b, c, d, e, f = y
    a_prime = (n_th*gamma*b**2)/4 - D*b*c + (n_th*gamma*c**2)/4 + gamma - f*D + (d*n_th*gamma)/2 + (e*n_th*gamma)/2
    b_prime = 4*D*c + (b*gamma)/2 - 2*D*c*d - D*b*f + b*d*gamma*n_th + (c*f*gamma*n_th)/2
    c_prime = (c*gamma)/2 - 2*D*b*e - D*c*f + (b*f*gamma*n_th)/2 + c*e*gamma*n_th
    d_prime = 4*D*f + d*gamma - 2*D*d*f + (gamma*n_th*(8*d**2 + 2*f**2))/8
    e_prime = e*gamma - 2*D*e*f + (gamma*n_th*(8*e**2 + 2*f**2))/8
    f_prime = 8*D*e + f*gamma - D*(f**2 + 4*d*e) + (gamma*n_th*(4*d*f + 4*e*f))/4
 

    return np.array([a_prime, b_prime, c_prime, d_prime,e_prime,f_prime])

# Analytical solution function
def ProbDensMap(x, y, solution):
    X, Y = np.meshgrid(x, y)
    a, b, c, d, e, f = solution
    ProbDens = np.exp(a + b * X + c * Y + d * X**2 + e * Y**2 + f * X * Y)
    return ProbDens

# Initialize simulation parameters
#   Physical Parameter
gamma = 0.1
D = 0.01
n_th = 0.0001

eps2 = 1
x0 = 2
y0 = 2

phys_parameter = gamma, D, n_th

Dim = 2
a0 = - (x0 * x0 + y0 * y0)/eps2  - Dim * np.log( np.sqrt(np.pi * eps2)) # Initial condition for a
b0 = 2 * x0 /eps2  # Initial condition for b
c0 = 2 * y0 /eps2  # Initial condition for c
d0 = - 1/eps2  # Initial condition for d
e0 = - 1/eps2  # Initial condition for d
f0 = 0
init_cond = np.array([a0, b0, c0, d0, e0, f0])

#   Time parameter
t_start = 0
t_end = 100
dt= 0.01

#   Map Parameter
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

output_dir = "PhotonSqueezing"

# Instantiate and run the simulation
simulator = FokkerPlanckSimulator(t_start, t_end, dt, x, y, phys_parameter, init_cond, output_dir, ProbDensMap, rk4_step)
simulator.run_simulation(pure_parameter=False)

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
    a, b, c, d, e, f, h, k, l, o, p, q = y
    a_prime = (n_th*gamma*b**2)/4 - D*b*c + (n_th*gamma*c**2)/4 + gamma - f*D + (d*n_th*gamma)/2 + (e*n_th*gamma)/2 - (k**2*n_th*gamma)/4 - (l**2*n_th*gamma)/4 + k*l*D
    b_prime = 4*D*c + (b*gamma)/2 - 2*D*c*d - D*b*f + 2*D*l*o + D*k*q - gamma*k*n_th*o - (gamma*l*n_th*q)/2 + b*d*gamma*n_th + (c*f*gamma*n_th)/2
    c_prime = (c*gamma)/2 - 2*D*b*e - D*c*f + 2*D*k*p + D*l*q - (gamma*k*n_th*q)/2 - gamma*l*n_th*p + (b*f*gamma*n_th)/2 + c*e*gamma*n_th
    d_prime = n_th*gamma*d**2 - 2*D*d*f + gamma*d + (n_th*gamma*f**2)/4 + 4*D*f - n_th*gamma*o**2 + 2*q*D*o - (n_th*q**2*gamma)/4
    e_prime = n_th*gamma*e**2 - 2*D*e*f + gamma*e + (n_th*gamma*f**2)/4 - n_th*gamma*p**2 + 2*q*D*p - (n_th*q**2*gamma)/4
    f_prime = 8*D*e + f*gamma - D*f**2 + D*q**2 - 4*D*d*e + 4*D*o*p + e*f*gamma*n_th - gamma*n_th*o*q - gamma*n_th*p*q + d*f*gamma*n_th

    h_prime = (gamma*n_th*o)/2 - D*b*l - D*c*k - D*q + (gamma*n_th*p)/2 + (b*gamma*k*n_th)/2 + (c*gamma*l*n_th)/2
    k_prime = 4*D*l + (gamma*k)/2 - 2*D*d*l - D*f*k - 2*D*c*o - D*b*q + d*gamma*k*n_th + b*gamma*n_th*o + (f*gamma*l*n_th)/2 + (c*gamma*n_th*q)/2
    l_prime = (gamma*l)/2 - 2*D*e*k - 2*D*b*p - D*f*l - D*c*q + e*gamma*l*n_th + (f*gamma*k*n_th)/2 + (b*gamma*n_th*q)/2 + c*gamma*n_th*p
    o_prime = 4*D*q + gamma*o - 2*D*d*q - 2*D*f*o + 2*d*gamma*n_th*o + (f*gamma*n_th*q)/2
    p_prime = gamma*p - 2*D*e*q - 2*D*f*p + 2*e*gamma*n_th*p + (f*gamma*n_th*q)/2
    q_prime = 8*D*p + gamma*q - 4*D*d*p - 4*D*e*o - 2*D*f*q + d*gamma*n_th*q + f*gamma*n_th*o + e*gamma*n_th*q + f*gamma*n_th*p

    return np.array([a_prime, b_prime, c_prime, d_prime,e_prime,f_prime, h_prime, k_prime,l_prime,o_prime, p_prime, q_prime])

# Analytical solution function
def ProbDensMap(x, y, solution):
    X, Y = np.meshgrid(x, y)
    a, b, c, d, e, f, h, k, l, o, p, q= solution
    ProbDens = np.exp(a + b * X + c * Y + d * X**2 + e * Y**2 + f * X * Y) * np.cos(h + k * X + l * Y + o * X**2 + p * Y**2 + q * X * Y)
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
f0 = 0.

h0 = 0.00
k0 = 0.
l0 = 0.
o0 = 0
p0 = 0
q0 = 0
init_cond = np.array([a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0])

#   Time parameter
t_start = 0
t_end = 10
dt= 0.01

#   Map Parameter
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

output_dir = "NewPhotonSqueezing"

# Instantiate and run the simulation
simulator = FokkerPlanckSimulator(t_start, t_end, dt, x, y, phys_parameter, init_cond, output_dir, ProbDensMap, rk4_step)
simulator.run_simulation(pure_parameter = True)

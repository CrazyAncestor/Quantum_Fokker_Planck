import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import os  # For handling directories
from fokker_planck_simulator import FokkerPlanckSimulator
from Symbolic_Time_Deriv_Solver import SymbolicSolver
import sympy as sp

gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym = sp.symbols('gamma_sym eta_sym nu_sym omega_sym n_th_sym m_th_sym')

# Use exponential function to simulate the solution of Fokker-Planck eq.
def PT_model_PREP(x,y,u,v,Px,Py,Pu,Pv,Pxx,Pyy,Puu,Pvv,Pxy,Puv,Pxu,Pyu,Pxv,Pyv):
    PT1 = (gamma + eta) + (gamma/2 * x - nu * y - g * v) * Px + (gamma/2 * y + (nu) * x + g * u) * Py
    PT2 = (eta/2 * u - (omega) * v - g *y) * Pu + (eta/2 * v + (omega) * u + g * x) * Pv
    PT3 = gamma * n_th / 4 * (Pxx + Pyy) + eta * m_th / 4 * (Puu + Pvv)
    return PT1 + PT2 + PT3 

x_sym, y_sym, u_sym, v_sym, a, b, c, d, e, f, h, k, l, o, p, q, r, s, w = sp.symbols('x_sym y_sym u_sym v_sym a b c d e f h k l o p q r s w')
P = sp.exp(a + b*x_sym + c*y_sym + d*u_sym + e*v_sym + f*x_sym**2 + h*y_sym**2 + k*u_sym**2 + l*v_sym**2 + o*x_sym*y_sym + p*u_sym*v_sym +q*x_sym*u_sym + r*y_sym*u_sym + s*x_sym*v_sym +w*y_sym*v_sym)

def generate_time_deriv_funcs(symsolver, PT_model):  
      
    PT_const, PTx, PTy, PTu, PTv, PTxx, PTyy, PTuu, PTvv, PTxy, PTuv, PTxu, PTyu, PTxv, PTyv = symsolver.two_mode_fokker_planck(PT_model, P,x_sym, y_sym, u_sym, v_sym, a, b, c, d, e, f, h, k, l, o, p, q, r, s, w)

    a_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym), PT_const, 'numpy')

    b_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym), PTx, 'numpy')
    c_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym), PTy, 'numpy')
    d_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym), PTu, 'numpy')
    e_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym), PTv, 'numpy')

    f_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym), PTxx, 'numpy')
    h_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym), PTyy, 'numpy')
    k_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym), PTuu, 'numpy')
    l_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym), PTvv, 'numpy')

    o_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym), PTxy, 'numpy')
    p_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym), PTuv, 'numpy')
    q_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym), PTxu, 'numpy')
    r_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym), PTyu, 'numpy')
    s_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym), PTxv, 'numpy')
    w_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym), PTyv, 'numpy')

    time_deriv_funcs = a_prime, b_prime, c_prime, d_prime,e_prime, f_prime, h_prime, k_prime,l_prime,o_prime, p_prime, q_prime, r_prime,s_prime,w_prime

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
    gamma, eta, nu, omega, n_th, m_th =  phys_parameter
    a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0 = y
    a_prime, b_prime, c_prime, d_prime,e_prime, f_prime, h_prime, k_prime,l_prime,o_prime, p_prime, q_prime, r_prime,s_prime,w_prime = time_deriv_funcs
    
    a_prime0 = a_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th)

    b_prime0 = b_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th)
    c_prime0 = c_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th)
    d_prime0 = d_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th)
    e_prime0 = e_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th)

    f_prime0 = f_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th)
    h_prime0 = h_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th)
    k_prime0 = k_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th)
    l_prime0 = l_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th)

    o_prime0 = o_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th)
    p_prime0 = p_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th)
    q_prime0 = q_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th)
    r_prime0 = r_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th)
    s_prime0 = s_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th)
    w_prime0 = w_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th)
    
    return np.array([a_prime0, b_prime0, c_prime0, d_prime0,e_prime0, f_prime0, h_prime0, k_prime0,l_prime0,o_prime0, p_prime0, q_prime0, r_prime0,s_prime0,w_prime0])

# Plotting complex representation function with parameters
def ProbDensMap(x, y, solution):
    X, Y = np.meshgrid(x, y)
    a, b, c, d, e, f, h, k, l, o, p, q, r, s, w = solution
    A = d + q * X + r * Y
    B = e + s * X + w * Y
    H = l - p**2/4/k
    U = B - A*p/2/k
    Prob0 = np.exp(f*X**2 + o*X*Y + b*X + h*Y**2 + c*Y + a)
    Prob_var = np.pi/ (np.abs(k)*np.abs(l-p**2/4/k))**0.5 *np.exp(-U**2/4/H)*np.exp(-A**2/4/k)
    ProbDens = Prob0 * Prob_var
    return ProbDens

def intensity(a,astar):
    return a*astar

# Initialize simulation parameters
#   Choice of Probability Representation
representation = 'P'

#   Physical Parameter
gamma = 0.5
eta = 0.5
nu = 2 * np.pi
omega = 2 * np.pi
g =  2 * np.pi * 0.1
n_th = 2
m_th = 2

eps2 = 1
x0 = 2
y0 = 2

phys_parameter = gamma, eta, nu, omega, n_th, m_th

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

#   Map Parameter
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

output_dir = "P_rep_RWA"

# Instantiate and run the simulation
SymSolver = SymbolicSolver()
time_deriv_funcs = generate_time_deriv_funcs(SymSolver, PT_model_PREP)
simulator = FokkerPlanckSimulator(representation, t_start, t_end, dt, x, y, phys_parameter, init_cond, output_dir, ProbDensMap, rk4_step, time_deriv_funcs)
simulator.run_simulation(pure_parameter = True)
simulator.electric_field_evolution()
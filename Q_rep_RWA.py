import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import os  # For handling directories
from fokker_planck_simulator import FokkerPlanckSimulator
from Symbolic_Time_Deriv_Solver import SymbolicSolver
import sympy as sp

def generate_time_deriv_funcs(symsolver, PHYS_model, prob_func_sym):  
      
    TD_const, TDx, TDy, TDu, TDv, TDxx, TDyy, TDuu, TDvv, TDxy, TDuv, TDxu, TDyu, TDxv, TDyv = symsolver.two_mode_fokker_planck(PHYS_model,prob_func_sym,x_sym, y_sym, u_sym, v_sym, a, b, c, d, e, f, h, k, l, o, p, q, r, s, w)

    a_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym), TD_const, 'numpy')

    b_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym), TDx, 'numpy')
    c_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym), TDy, 'numpy')
    d_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym), TDu, 'numpy')
    e_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym), TDv, 'numpy')

    f_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym), TDxx, 'numpy')
    h_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym), TDyy, 'numpy')
    k_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym), TDuu, 'numpy')
    l_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym), TDvv, 'numpy')

    o_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym), TDxy, 'numpy')
    p_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym), TDuv, 'numpy')
    q_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym), TDxu, 'numpy')
    r_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym), TDyu, 'numpy')
    s_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym), TDxv, 'numpy')
    w_prime = sp.lambdify((a, b, c, d, e, f, h, k, l, o, p, q, r, s, w,  gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym), TDyv, 'numpy')

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
    gamma, eta, nu, omega, n_th, m_th, g =  phys_parameter
    a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0 = y

    a_prime, b_prime, c_prime, d_prime,e_prime, f_prime, h_prime, k_prime,l_prime,o_prime, p_prime, q_prime, r_prime,s_prime,w_prime = time_deriv_funcs
    
    a_prime0 = a_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)

    b_prime0 = b_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)
    c_prime0 = c_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)
    d_prime0 = d_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)
    e_prime0 = e_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)

    f_prime0 = f_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)
    h_prime0 = h_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)
    k_prime0 = k_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)
    l_prime0 = l_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)

    o_prime0 = o_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)
    p_prime0 = p_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)
    q_prime0 = q_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)
    r_prime0 = r_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)
    s_prime0 = s_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)
    w_prime0 = w_prime(a0, b0, c0, d0, e0, f0, h0, k0, l0, o0, p0, q0, r0, s0, w0, gamma, eta, nu, omega, n_th, m_th, g)
    
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

#   Map Parameter
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

#   Output directory
output_dir = "Q_rep_RWA"

# Symbolic formula solver settings
#   Choice of Probability Representation
representation = 'Q'
SymSolver = SymbolicSolver()
gamma_sym, eta_sym, nu_sym, omega_sym, n_th_sym, m_th_sym, g_sym = sp.symbols('gamma_sym eta_sym nu_sym omega_sym n_th_sym m_th_sym g_sym')

def RWA_QREP(x,y,u,v,Px,Py,Pu,Pv,Pxx,Pyy,Puu,Pvv,Pxy,Puv,Pxu,Pyu,Pxv,Pyv):
    TD1 = (gamma_sym + eta_sym) + (gamma_sym/2 * x - nu_sym * y - g_sym * v) * Px + (gamma_sym/2 * y + (nu_sym) * x + g_sym * u) * Py
    TD2 = (eta_sym/2 * u - (omega_sym) * v - g_sym *y) * Pu + (eta_sym/2 * v + (omega_sym) * u + g_sym * x) * Pv
    TD3 = gamma_sym * (n_th_sym+1) / 4 * (Pxx + Pyy) + eta_sym * (m_th_sym+1) / 4 * (Puu + Pvv)
    return TD1 + TD2 + TD3  

x_sym, y_sym, u_sym, v_sym, a, b, c, d, e, f, h, k, l, o, p, q, r, s, w = sp.symbols('x_sym y_sym u_sym v_sym a b c d e f h k l o p q r s w')

# Use exponential function to simulate the solution of Fokker-Planck eq.
ProbFunc = sp.exp(a + b*x_sym + c*y_sym + d*u_sym + e*v_sym + f*x_sym**2 + h*y_sym**2 + k*u_sym**2 + l*v_sym**2 + o*x_sym*y_sym + p*u_sym*v_sym +q*x_sym*u_sym + r*y_sym*u_sym + s*x_sym*v_sym +w*y_sym*v_sym)

time_deriv_funcs = generate_time_deriv_funcs(SymSolver, RWA_QREP, ProbFunc)

# Instantiate and run the simulation
simulator = FokkerPlanckSimulator(representation, t_start, t_end, dt, x, y, phys_parameter, init_cond, output_dir, ProbDensMap, rk4_step, time_deriv_funcs)
simulator.run_simulation(pure_parameter = False)
simulator.electric_field_evolution()
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cmath
import sympy as sp

class SymbolicSolver:
    def __init__(self):
        # Simulation time settings
        self.var = 0
    
    # 一階導數
    def fst_deriv(self, func, var):
        return sp.simplify(sp.diff(func, var))

    # 二階導數
    def snd_deriv(self, func, var):
        return sp.simplify(sp.diff(func, var, var))

    # 混合二階導數
    def diff_deriv(self, func, var1, var2):
        return sp.simplify(sp.diff(sp.diff(func, var1), var2))

    # 指數函數的一階導數
    def exp_fst_deriv(self, func, var):
        return sp.simplify(sp.diff(func, var) / func)

    # 指數函數的二階導數
    def exp_snd_deriv(self, func, var):
        return sp.simplify(sp.diff(sp.diff(func, var), var) / func)

    # 指數函數的混合二階導數
    def exp_diff_deriv(self, func, var1, var2):
        return sp.simplify(sp.diff(sp.diff(func, var1), var2) / func)
    
    def one_mode_fokker_planck(self, PT_model,func,x, y, a, b, c, d, e, f):
            P = sp.exp(a + b*x + c*y + d*x**2 + e*y**2 + f*x*y)
            
            # 計算 P 的一階和二階導數
            def fst_snd_derivs(func, x, y):
                Px = self.exp_fst_deriv(func, x)
                Py = self.exp_fst_deriv(func, y)

                Pxx = self.exp_snd_deriv(func, x)
                Pyy = self.exp_snd_deriv(func, y)

                Pxy = self.exp_diff_deriv(func,x,y)
                
                return Px, Py, Pxx, Pyy, Pxy

            # 計算對 x 和 y 的導數
            Px, Py, Pxx, Pyy, Pxy = fst_snd_derivs(func, x, y)

            # 計算 P 的時間導數
            PT = PT_model(x,y,Px,Py,Pxx,Pyy,Pxy)

            # 提取係數的函數
            def extract_coef(PT, x, y):
                PTxx = self.snd_deriv(PT, x)
                PTyy = self.snd_deriv(PT, y)
                PTxy = self.diff_deriv(PT,x,y)

                PTx = sp.simplify(self.fst_deriv(PT, x) - (PTxx * x) - (PTxy * y))
                PTy = sp.simplify(self.fst_deriv(PT, y) - (PTyy * y) - (PTxy * x))

                PT_const = PT - ((PTxx * x**2 + PTyy * y**2) / 2 + PTxy * x * y + (PTx * x + PTy * y))
                PT_const = sp.simplify(PT_const)

                PTxx = PTxx / 2
                PTyy = PTyy / 2
                
                return PT_const, PTx, PTy, PTxx, PTyy, PTxy

            # 提取時間導數的各項係數
            
            PT_const, PTx, PTy, PTxx, PTyy, PTxy = extract_coef(PT, x, y)
            return PT_const, PTx, PTy, PTxx, PTyy, PTxy
    
    def two_mode_fokker_planck(self, PT_model, func,x, y, u, v, a, b, c, d, e, f, h, k, l, o, p, q, r, s, w):
            
            # Calculate the first and second derivatives for P
            def fst_snd_derivs(func, x, y, u, v):
                Px = self.exp_fst_deriv(func, x)
                Py = self.exp_fst_deriv(func, y)
                Pu = self.exp_fst_deriv(func, u)
                Pv = self.exp_fst_deriv(func, v)
                
                Pxx = self.exp_snd_deriv(func, x)
                Pyy = self.exp_snd_deriv(func, y)
                Puu = self.exp_snd_deriv(func, u)
                Pvv = self.exp_snd_deriv(func, v)

                Pxy = self.exp_diff_deriv(func, x, y)
                Puv = self.exp_diff_deriv(func, u, v)
                Pxu = self.exp_diff_deriv(func, x, u)
                Pyu = self.exp_diff_deriv(func, y, u)
                Pxv = self.exp_diff_deriv(func, x, v)
                Pyv = self.exp_diff_deriv(func, y, v)
                
                return Px, Py, Pu, Pv, Pxx, Pyy, Puu, Pvv, Pxy, Puv, Pxu, Pyu, Pxv, Pyv
    
             # Calculate first and second derivatives
            Px, Py, Pu, Pv, Pxx, Pyy, Puu, Pvv, Pxy, Puv, Pxu, Pyu, Pxv, Pyv = fst_snd_derivs(func, x, y, u, v)


            # 計算 P 的時間導數
            PT = PT_model(x,y,u,v,Px,Py,Pu,Pv,Pxx,Pyy,Puu,Pvv,Pxy,Puv,Pxu,Pyu,Pxv,Pyv)

            # Extract each component in the time derivative
            def extract_coef(PT, x, y, u, v):
                PTxx = self.snd_deriv(PT, x)
                PTyy = self.snd_deriv(PT, y)
                PTuu = self.snd_deriv(PT, u)
                PTvv = self.snd_deriv(PT, v)
                
                PTxy = self.diff_deriv(PT, x, y)
                PTuv = self.diff_deriv(PT, u, v)
                PTxu = self.diff_deriv(PT, x, u)
                PTyu = self.diff_deriv(PT, y, u)
                PTxv = self.diff_deriv(PT, x, v)
                PTyv = self.diff_deriv(PT, y, v)
                
                PTx = sp.simplify(self.fst_deriv(PT, x) - (PTxx * x + PTxu * u + PTxv * v + PTxy * y))
                PTy = sp.simplify(self.fst_deriv(PT, y) - (PTyy * y + PTyu * u + PTyv * v + PTxy * x))
                
                PTu = sp.simplify(self.fst_deriv(PT, u) - (PTuu * u + PTxu * x + PTuv * v + PTyu * y))
                PTv = sp.simplify(self.fst_deriv(PT, v) - (PTvv * v + PTuv * u + PTyv * y + PTxv * x))
                
                PT_const = PT - ((PTxx * x**2 + PTyy * y**2 + PTuu * u**2 + PTvv * v**2)/2 + 
                                (PTxu * x * u + PTyu * y * u + PTxv * x * v + PTyv * y * v + PTxy * x * y + PTuv * u * v) + 
                                (PTx * x + PTy * y + PTu * u + PTv * v))
                PT_const = sp.simplify(PT_const)

                PTxx = PTxx / 2
                PTyy = PTyy / 2
                PTuu = PTuu / 2
                PTvv = PTvv / 2
                
                return PT_const, PTx, PTy, PTu, PTv, PTxx, PTyy, PTuu, PTvv, PTxy, PTuv, PTxu, PTyu, PTxv, PTyv


            # 提取時間導數的各項係數
            
            # Extract coefficients from the time-derivative
            PT_const, PTx, PTy, PTu, PTv, PTxx, PTyy, PTuu, PTvv, PTxy, PTuv, PTxu, PTyu, PTxv, PTyv = extract_coef(PT, x, y, u, v)
            return PT_const, PTx, PTy, PTu, PTv, PTxx, PTyy, PTuu, PTvv, PTxy, PTuv, PTxu, PTyu, PTxv, PTyv
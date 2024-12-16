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
    
    def one_mode_fokker_planck(self, PHYS_model,func,x, y, a, b, c, d, e, f):
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
            TD = PHYS_model(x,y,Px,Py,Pxx,Pyy,Pxy)

            # 提取係數的函數
            def extract_coef(TD, x, y):
                TDxx = self.snd_deriv(TD, x)
                TDyy = self.snd_deriv(TD, y)
                TDxy = self.diff_deriv(TD,x,y)

                TDx = sp.simplify(self.fst_deriv(TD, x) - (TDxx * x) - (TDxy * y))
                TDy = sp.simplify(self.fst_deriv(TD, y) - (TDyy * y) - (TDxy * x))

                TD_const = TD - ((TDxx * x**2 + TDyy * y**2) / 2 + TDxy * x * y + (TDx * x + TDy * y))
                TD_const = sp.simplify(TD_const)

                TDxx = TDxx / 2
                TDyy = TDyy / 2
                
                return TD_const, TDx, TDy, TDxx, TDyy, TDxy

            # 提取時間導數的各項係數
            
            TD_const, TDx, TDy, TDxx, TDyy, TDxy = extract_coef(TD, x, y)
            return TD_const, TDx, TDy, TDxx, TDyy, TDxy
    
    def two_mode_fokker_planck(self, PHYS_model, func,x, y, u, v, a, b, c, d, e, f, h, k, l, o, p, q, r, s, w):
            
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
            TD = PHYS_model(x,y,u,v,Px,Py,Pu,Pv,Pxx,Pyy,Puu,Pvv,Pxy,Puv,Pxu,Pyu,Pxv,Pyv)

            # Extract each component in the time derivative
            def extract_coef(TD, x, y, u, v):
                TDxx = self.snd_deriv(TD, x)
                TDyy = self.snd_deriv(TD, y)
                TDuu = self.snd_deriv(TD, u)
                TDvv = self.snd_deriv(TD, v)
                
                TDxy = self.diff_deriv(TD, x, y)
                TDuv = self.diff_deriv(TD, u, v)
                TDxu = self.diff_deriv(TD, x, u)
                TDyu = self.diff_deriv(TD, y, u)
                TDxv = self.diff_deriv(TD, x, v)
                TDyv = self.diff_deriv(TD, y, v)
                
                TDx = sp.simplify(self.fst_deriv(TD, x) - (TDxx * x + TDxu * u + TDxv * v + TDxy * y))
                TDy = sp.simplify(self.fst_deriv(TD, y) - (TDyy * y + TDyu * u + TDyv * v + TDxy * x))
                
                TDu = sp.simplify(self.fst_deriv(TD, u) - (TDuu * u + TDxu * x + TDuv * v + TDyu * y))
                TDv = sp.simplify(self.fst_deriv(TD, v) - (TDvv * v + TDuv * u + TDyv * y + TDxv * x))
                
                TD_const = TD - ((TDxx * x**2 + TDyy * y**2 + TDuu * u**2 + TDvv * v**2)/2 + 
                                (TDxu * x * u + TDyu * y * u + TDxv * x * v + TDyv * y * v + TDxy * x * y + TDuv * u * v) + 
                                (TDx * x + TDy * y + TDu * u + TDv * v))
                TD_const = sp.simplify(TD_const)

                TDxx = TDxx / 2
                TDyy = TDyy / 2
                TDuu = TDuu / 2
                TDvv = TDvv / 2
                
                return TD_const, TDx, TDy, TDu, TDv, TDxx, TDyy, TDuu, TDvv, TDxy, TDuv, TDxu, TDyu, TDxv, TDyv


            # 提取時間導數的各項係數
            
            # Extract coefficients from the time-derivative
            TD_const, TDx, TDy, TDu, TDv, TDxx, TDyy, TDuu, TDvv, TDxy, TDuv, TDxu, TDyu, TDxv, TDyv = extract_coef(TD, x, y, u, v)
            return TD_const, TDx, TDy, TDu, TDv, TDxx, TDyy, TDuu, TDvv, TDxy, TDuv, TDxu, TDyu, TDxv, TDyv
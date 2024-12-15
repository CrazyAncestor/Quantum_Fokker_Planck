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
    
    def solve_fkpk_symbol(self, PT_model,x, y, a, b, c, d, e, f):
            # 定義符號變數
            

            # 定義函數 P(x, y)
            P = sp.exp(a + b*x + c*y + d*x**2 + e*y**2 + f*x*y)

            # 一階導數
            def fst_deriv(func, var):
                return sp.simplify(sp.diff(func, var))

            # 二階導數
            def snd_deriv(func, var):
                return sp.simplify(sp.diff(func, var, var))

            # 混合二階導數
            def diff_deriv(func, var1, var2):
                return sp.simplify(sp.diff(sp.diff(func, var1), var2))

            # 指數函數的一階導數
            def exp_fst_deriv(func, var):
                return sp.simplify(sp.diff(func, var) / func)

            # 指數函數的二階導數
            def exp_snd_deriv(func, var):
                return sp.simplify(sp.diff(sp.diff(func, var), var) / func)

            # 指數函數的混合二階導數
            def exp_diff_deriv(func, var1, var2):
                return sp.simplify(sp.diff(sp.diff(func, var1), var2) / func)

            # 計算 P 的一階和二階導數
            def fst_snd_derivs(P, x, y):
                Px = exp_fst_deriv(P, x)
                Py = exp_fst_deriv(P, y)

                Pxx = exp_snd_deriv(P, x)
                Pyy = exp_snd_deriv(P, y)

                Pxy = exp_diff_deriv(P,x,y)
                
                return Px, Py, Pxx, Pyy, Pxy

            # 計算對 x 和 y 的導數
            Px, Py, Pxx, Pyy, Pxy = fst_snd_derivs(P, x, y)

            # 計算 P 的時間導數
            PT = PT_model(x,y,Px,Py,Pxx,Pyy,Pxy)

            # 提取係數的函數
            def extract_coef(PT, x, y):
                PTxx = snd_deriv(PT, x)
                PTyy = snd_deriv(PT, y)
                PTxy = diff_deriv(PT,x,y)

                PTx = sp.simplify(fst_deriv(PT, x) - (PTxx * x) - (PTxy * y))
                PTy = sp.simplify(fst_deriv(PT, y) - (PTyy * y) - (PTxy * x))

                PT_const = PT - ((PTxx * x**2 + PTyy * y**2) / 2 + PTxy * x * y + (PTx * x + PTy * y))
                PT_const = sp.simplify(PT_const)

                PTxx = PTxx / 2
                PTyy = PTyy / 2
                
                return PT_const, PTx, PTy, PTxx, PTyy, PTxy

            # 提取時間導數的各項係數
            
            PT_const, PTx, PTy, PTxx, PTyy, PTxy = extract_coef(PT, x, y)
            return PT_const, PTx, PTy, PTxx, PTyy, PTxy
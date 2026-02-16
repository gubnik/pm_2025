#!/bin/python3

import inspect
from jinja2 import Environment, FileSystemLoader
import sys
import math
from typing import Callable
import numpy as np
import sympy as sp

import printer

points = ['A', 'B', 'C', 'D', 'E', 'F']

def min_in_range(f: Callable, test_range: np.ndarray, **kwargs):
    return min([f(i, **kwargs) for i in test_range])

def constants():
    g = 9.81
    rho = 1100
    H = 30
    Q = 1e-4
    lambda_fr = 0.010
    L_pipe = 3
    D_pipe = 25.4 * 1e-3
    v_flow = 4 * Q / (np.pi * D_pipe**2)
    lambda_len = 3
    n = 3
    z = 5
    eta_v = 0.9
    eta_Sigma = 0.6
    return locals()

def equations():
    Q, rho, g, H = sp.symbols('Q rho g H')
    P = rho * g * H
    lambda_fr, L_pipe, D_pipe, v_flow = sp.symbols('lambda_fr L_pipe D_pipe v_flow')
    Delta_P = lambda_fr * L_pipe * rho * (v_flow ** 2) / (2 * D_pipe)
    eta_Sigma = sp.symbols('eta_Sigma')
    R, L = sp.symbols('R L')
    t = sp.symbols('t')
    phi_f = sp.Function('phi')
    phi = phi_f(t)
    x_A = R*0
    y_A = R*0
    x_B = R * sp.cos(phi)
    y_B = R * sp.sin(phi)
    x_C = R * sp.cos(phi) + sp.sqrt(L**2 - (R**2) * sp.sin(phi)**2)
    y_C = R*0
    x_D = R*0
    y_D = R*0
    x_E = R * sp.cos(phi) / 2
    y_E = R * sp.sin(phi) / 2
    x_F = R * sp.cos(phi) + sp.sqrt(L**2 - (R**2) * sp.sin(phi)**2) / 2
    y_F = R * sp.sin(phi) / 2

    vels = {}
    omega = 5 * sp.pi * 2
    for p in points:
        vel_xp = sp.diff(locals()[f'x_{p}'], t)
        vel_yp = sp.diff(locals()[f'y_{p}'], t)
        vels[f'v_x_{p}'] = vel_xp
        vels[f'v_y_{p}'] = vel_yp

    d, z, n, eta_v, lamb = sp.symbols('d z n eta_v lambda_len')
    A = sp.pi * (d**2) / 4
    R = Q  / (2 * n * z * A * eta_v)
    S = 2 * R
    L = lamb * R
    F = P * A
    N = Q * H * rho * g
    N_used = N / eta_Sigma

    return {'t': t, 'phi': phi,
            'P': P, 'Delta_P': Delta_P,
            'x_A': x_A, 'y_A': y_A,
            'x_B': x_B, 'y_B': y_B,
            'x_C': x_C, 'y_C': y_C,
            'x_D': x_D, 'y_D': y_D,
            'x_E': x_E, 'y_E': y_E,
            'x_F': x_F, 'y_F': y_F,
            'R': R, 'A': A, 'S': S, 'L': L, 'F': F, 
            'N': N, 'N_used': N_used,
            **vels}

def substitutes(consts, eqs):
    return {f"{name}_val": eq.subs(consts).evalf() for name, eq in eqs.items()}

print(substitutes(constants(), equations()))

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        exit(1)

    authors = [
            "Бабенкова Е.А.",
            "Губанков Н.Г.",
            "Калинина А.В.",
            "Чабанов А.Р",
            "Швайковский Е.А.",
            "Дворкин Ф.В.",
            ]
   
    env = Environment(
            loader=FileSystemLoader("."),
            autoescape=False
            )

    as_fun = lambda symb_str, expr: sp.lambdify(sp.symbols(symb_str), expr)
    env.globals.update({
        'latex': sp.latex,
        'diff': sp.diff,
        'subs': sp.Expr.subs,
        'as_fun': as_fun,
        'symb': sp.symbols,
        'mul': sp.Mul,
        'phi_print': printer.phi_print,
        'globals': globals
        })
    
    template = env.get_template(sys.argv[1])
    
    ctx = {
        **constants(),
        **equations(),
        **substitutes(constants(), equations()),
        }

    rendered = template.render(
            points=points,
            ctx=ctx,
            **ctx,
            )
    
    out = sys.argv[2]
    
    with open(out, "w", encoding="utf-8") as f:
        f.write(rendered)
    
    print("Render completed!")

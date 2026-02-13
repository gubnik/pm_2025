#!/bin/python3

from jinja2 import Environment, FileSystemLoader
import sys
import math
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def min_in_range(f: Callable, test_range: np.ndarray, **kwargs):
    return min([f(i, **kwargs) for i in test_range])

def q_fn(phi: float) -> float:
    return max(0, math.sin(phi))

def q_Sigma_fn(phi: float, n: int) -> float:
    return sum(q_fn(phi + (2 * math.pi * k)/n) for k in range(0, n))

def q_Norm_fn(phi: float, n: int) -> float:
    return q_Sigma_fn(phi, n) / q_Sigma_fn(n * np.pi / 2, n)

def plot_to_png(filename: str, f: Callable, test_range: np.ndarray, x_label: str, y_label: str, do_pi_labels: bool = False, **kwargs):
    vals = [f(i, **kwargs) for i in test_range]
    plt.figure(figsize=(10, 5))
    plt.plot(test_range, vals, color='blue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if do_pi_labels:
        plt.xticks(np.arange(test_range.min(), test_range.max(), np.pi / 2),
                   [f'{i/2}π' if i != 0 else '0' for i in range(int(test_range.max() / (np.pi/2)) + 1)])
    plt.axhline(0, color='gray', lw=1, ls='--')
    plt.axvline(0, color='gray', lw=1, ls='--')
    plt.grid()
    plt.savefig(filename)
    plt.close()

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
    d, z, n, eta_v, lamb = sp.symbols('d z n eta_v lambda_len')
    A = sp.pi * (d**2) / 4
    R = Q  / (2 * n * z * A * eta_v)
    S = 2 * R
    L = lamb * R
    F = P * A
    N = Q * H * rho * g
    N_used = N / eta_Sigma
    return {'P': P, 'Delta_P': Delta_P, 'R': R, 'A': A, 'S': S, 'L': L, 'F': F, 
            'N': N, 'N_used': N_used}

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

    q_Sigma_min = min_in_range(q_Norm_fn, np.arange(0, 2 * math.pi, 0.01), n = 3)
    for k in [1, 3, 5, 7]:
        plot_to_png(f"img/pulse_{k}.png", lambda phi: q_Norm_fn(phi, k), np.arange(0, 4 * np.pi, 0.01),
                    x_label="Фаза цилиндра (phi)",
                    y_label="Относительный моментальный расход (%)",
                    do_pi_labels=True)
    
    env = Environment(
            loader=FileSystemLoader("."),
            autoescape=False
            )

    as_fun = lambda symb_str, expr: sp.lambdify(sp.symbols(symb_str), expr)
    env.globals.update({
        'latex': sp.latex,
        'diff': sp.diff,
        'as_fun': as_fun,
        'symb': sp.symbols,
        })
    
    template = env.get_template(sys.argv[1])
    
    rendered = template.render(
            **constants(),
            **equations(),
            **substitutes(constants(), equations())
            )
    
    out = sys.argv[2]
    
    with open(out, "w", encoding="utf-8") as f:
        f.write(rendered)
    
    print("Render completed!")

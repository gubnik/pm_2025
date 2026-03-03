import math
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from main import max_in_range, points
from main import constants, equations, substitutes
import main

def q_fn(phi: float) -> float:
    return max(0, math.sin(phi))

def q_Sigma_fn(phi: float, n: int) -> float:
    return sum(q_fn(phi + (2 * math.pi * k)/n) for k in range(0, n))

def q_Norm_fn(phi: float, n: int) -> float:
    return q_Sigma_fn(phi, n) / q_Sigma_fn(n * np.pi / 2, n)

def plot_to_png(filename: str, f: Callable, test_range: np.ndarray, x_label: str, y_label: str, do_pi_labels: bool = False):
    vals = [f(i) for i in test_range]
    plt.figure(figsize=(10, 5))
    plt.plot(test_range, vals, color='blue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if do_pi_labels:
        plt.xticks(np.arange(test_range.min(), test_range.max() + 1, np.pi / 2),
                   [f'{i/2}π' if i != 0 else '0' for i in range(int(test_range.max() / (np.pi/2)) + 2)])
    plt.axhline(0, color='gray', lw=1, ls='--')
    plt.axvline(0, color='gray', lw=1, ls='--')
    plt.grid()
    plt.savefig(filename)
    plt.close()

def single_acting_window(phi: float):
    return max(0, phi)

def plot_multiple(filename: str, functions: list[Callable], test_range: np.ndarray, x_label: str, y_label: str,
                  do_pi_labels: bool = False,
                  legend: list[str] = []):
    plt.figure(figsize=(10, 5))
    if len(legend) != 0 and len(legend) != len(functions):
        raise RuntimeError(f"legend is {len(legend)} params, functions are {len(functions)}")
    for i, f in enumerate(functions):
        vals = [f(i) for i in test_range]
        plt.plot(test_range, vals, label=legend[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if do_pi_labels:
        plt.xticks(np.arange(test_range.min(), test_range.max() + 1, np.pi / 2),
                   [f'{i/2}π' if i != 0 else '0' for i in range(int(test_range.max() / (np.pi/2)) + 2)])
    plt.legend()
    plt.axhline(0, color='gray', lw=1, ls='--')
    plt.axvline(0, color='gray', lw=1, ls='--')
    plt.grid()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    caption_phi ="Фаза цилиндра, радианы"
    cs = constants()
    eqs = equations()
    subs = substitutes(cs, eqs)
    q_Sigma_min = main.min_in_range(q_Norm_fn, np.arange(0, 2 * math.pi, 0.01), n = 3)
    for k in [1, 3, 5, 7]:
        plot_to_png(f"img/gen/pulse_{k}.png", lambda phi: q_Norm_fn(phi, k), np.arange(0, 2 * np.pi, 0.01),
                    x_label=caption_phi,
                    y_label="Относительный моментальный расход (%)",
                    do_pi_labels=True)
    d = sp.symbols('d')
    d_val = 19e-3
    omega = 5.0 * sp.pi * 2
    R_val = sp.lambdify(d, subs['R_val'])(d_val)
    L_val = sp.lambdify(d, subs['L_val'])(d_val)
    v_x_fs = []
    v_y_fs = []
    a_x_fs = []
    a_y_fs = []
    phi_symb = sp.symbols('phi')
    vels = {}
    for p in points:
        v_x = eqs[f'v_x_{p}'].xreplace({eqs['phi']: phi_symb, sp.diff(eqs['phi']) : omega}).subs(sp.symbols('R'), R_val).subs(sp.symbols('L'), L_val)
        v_y = eqs[f'v_y_{p}'].xreplace({eqs['phi']: phi_symb, sp.diff(eqs['phi']) : omega}).subs(sp.symbols('R'), R_val).subs(sp.symbols('L'), L_val)
        a_x = eqs[f'a_x_{p}'].xreplace({eqs['phi']: phi_symb, sp.diff(eqs['phi']) : omega}).subs(sp.symbols('R'), R_val).subs(sp.symbols('L'), L_val)
        a_y = eqs[f'a_y_{p}'].xreplace({eqs['phi']: phi_symb, sp.diff(eqs['phi']) : omega}).subs(sp.symbols('R'), R_val).subs(sp.symbols('L'), L_val)
        print(f'{p}: {sp.latex(v_x)}')
        v_x_f = sp.lambdify(phi_symb, v_x)
        v_x_fs.append(v_x_f)
        v_y_f = sp.lambdify(phi_symb, v_y)
        v_y_fs.append(v_y_f)
        a_x_f = sp.lambdify(phi_symb, a_x)
        a_x_fs.append(a_x_f)
        a_y_f = sp.lambdify(phi_symb, a_y)
        a_y_fs.append(a_y_f)
        vels[p] = lambda phi: math.sqrt(v_x_f(phi)**2 + v_y_f(phi)**2)
    plot_multiple(filename=f"img/gen/vel_x.png",
                  functions=v_x_fs,
                  legend=[f"Точка {p}" for p in points],
                  test_range=np.arange(0, 2 * np.pi, 0.01),
                  x_label=caption_phi,
                  y_label=f"v_x, скорость по оси x, м/с",
                  do_pi_labels=True)
    plot_multiple(filename=f"img/gen/vel_y.png",
                  functions=v_y_fs,
                  legend=[f"Точка {p}" for p in points],
                  test_range=np.arange(0, 2 * np.pi, 0.01),
                  x_label=caption_phi,
                  y_label=f"v_x, скорость по оси y, м/с",
                  do_pi_labels=True)
    plot_multiple(filename=f"img/gen/vel_mod.png",
                  functions=[lambda phi, v_x_f=v_x_f, v_y_f=v_y_f: math.sqrt(v_x_f(phi)**2 + v_y_f(phi)**2)
                             for v_x_f, v_y_f in zip(v_x_fs, v_y_fs)],
                  legend=[f"Точка {p}" for p in points],
                  test_range=np.arange(0, 2 * np.pi, 0.01),
                  x_label=caption_phi,
                  y_label=f"v_x, скорость по оси y, м/с",
                do_pi_labels=True)
    plot_multiple(filename=f"img/gen/acc_x.png",
                  functions=a_x_fs,
                  legend=[f"Точка {p}" for p in points],
                  test_range=np.arange(0, 2 * np.pi, 0.01),
                  x_label=caption_phi,
                  y_label=f"a_x, ускорение по оси x, м/с",
                  do_pi_labels=True)
    plot_multiple(filename=f"img/gen/acc_y.png",
                  functions=a_y_fs,
                  legend=[f"Точка {p}" for p in points],
                  test_range=np.arange(0, 2 * np.pi, 0.01),
                  x_label=caption_phi,
                  y_label=f"a_x, ускорение по оси y, м/с",
                  do_pi_labels=True)
    plot_multiple(filename=f"img/gen/acc_mod.png",
                  functions=[lambda phi, a_x_f=a_x_f, a_y_f=a_y_f: math.sqrt(a_x_f(phi)**2 + a_y_f(phi)**2)
                             for a_x_f, a_y_f in zip(a_x_fs, a_y_fs)],
                  legend=[f"Точка {p}" for p in points],
                  test_range=np.arange(0, 2 * np.pi, 0.01),
                  x_label=caption_phi,
                  y_label=f"a_x, ускорение по оси y, м/с",
                  do_pi_labels=True)
    M_A = subs[f'M_A_val'].xreplace({eqs['phi']: phi_symb, sp.diff(eqs['phi']) : omega,
                                d: d_val})
    M_A_f = sp.lambdify(phi_symb, M_A)
    plot_to_png(f"img/gen/moment_A.png", lambda phi: M_A_f(phi), np.arange(0, 2 * np.pi, 0.01),
                    x_label=caption_phi,
                    y_label="Приводной момент в точке A, Н * м",
                    do_pi_labels=True)
    print(max_in_range(M_A_f, np.arange(0, 2*np.pi, 0.01)))
    for p in ['A', 'B', 'C']:
        Z_x_f = sp.lambdify(phi_symb, subs[f'Z_{p}_x_val'].
                            xreplace({eqs['phi']: phi_symb,
                                      sp.diff(eqs['phi']) : omega,
                                      d: d_val}))
        Z_y_f = sp.lambdify(phi_symb, subs[f'Z_{p}_y_val'].
                            xreplace({eqs['phi']: phi_symb,
                                      sp.diff(eqs['phi']) : omega,
                                      d: d_val}))
        Z_f = sp.lambdify(phi_symb, subs[f'Z_{p}_val'].
                          xreplace({eqs['phi']: phi_symb,
                                    sp.diff(eqs['phi']) : omega,
                                    d: d_val}))
        plot_multiple(filename=f"img/gen/reactions_{p}.png",
                      functions=[lambda phi, f=f: f(phi) for f in [Z_x_f, Z_y_f, Z_f]],
                      legend=['OX', 'OY', 'Модуль'],
                      test_range=np.arange(0, 2 * np.pi, 0.01),
                      x_label=caption_phi,
                      y_label=f"Реакция в точке {p}, Н",
                      do_pi_labels=True)


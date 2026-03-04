#!/bin/python3

from jinja2 import Environment, FileSystemLoader
import sys
import math
from typing import Callable
import numpy as np
import sympy as sp
import scipy

import printer

points = ['A', 'B', 'C', 'D', 'E', 'F']
def min_in_range(f: Callable, test_range: np.ndarray, **kwargs):
    return min([f(i, **kwargs) for i in test_range])

def max_in_range(f: Callable, test_range: np.ndarray, **kwargs):
    return max([f(i, **kwargs) for i in test_range])

def constants():
    g = 9.81
    rho = 1050
    H = 50
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
    n_omega = 5
    omega = n_omega * np.pi * 2
    rho_m = 7850
    d = 19.05e-3
    h_r = 4.5e-3
    b_r = 4.5e-3
    R_raw = 13*1e-3
    S_raw = 2*R_raw
    L_raw = 3*R_raw
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
    for p in points:
        vel_xp = sp.diff(locals()[f'x_{p}'], t)
        vel_yp = sp.diff(locals()[f'y_{p}'], t)
        vels[f'v_x_{p}'] = vel_xp
        vels[f'v_y_{p}'] = vel_yp
    accs = {}
    for p in points:
        # \phi'' = 0 bc \phi' is const
        acc_xp = sp.diff(vels[f'v_x_{p}'], t).subs(sp.diff(sp.diff(phi)), 0)
        acc_yp = sp.diff(vels[f'v_y_{p}'], t).subs(sp.diff(sp.diff(phi)), 0)
        accs[f'a_x_{p}'] = acc_xp
        accs[f'a_y_{p}'] = acc_yp
        accs[f'a_{p}'] = sp.sqrt(acc_yp**2 + acc_xp**2)

    d, z, n, eta_v, lamb = sp.symbols('d z n eta_v lambda_len')
    b_r, h_r = sp.symbols('b_r h_r')

    A_p = sp.pi * (d**2) / 4
    W_p = sp.pi*d**3/32
    A_r = h_r * b_r
    W_r = b_r * (h_r**2) / 6

    R = sp.ceiling(1e+3 * (Q  / (2 * n * z * A_p * eta_v))) / 1e+3
    S = 2 * R
    L = lamb * R
    F = P * A_p
    N = Q * H * rho * g
    N_used = N / eta_Sigma

    sin_theta = R*sp.sin(phi) / L
    cos_theta = (sp.sqrt(L**2 - (R**2)*sp.sin(phi)**2)) / L
    tan_theta = sin_theta / cos_theta
    Z_C_x = F
    Z_C_y = F * tan_theta
    Z_B_x = -Z_C_x
    Z_B_y = -Z_C_y
    Z_A_x = Z_C_x
    Z_A_y = Z_C_y
    Z_C = sp.sqrt(Z_C_x**2 + Z_C_y**2)
    Z_B = sp.sqrt(Z_B_x**2 + Z_B_y**2)
    Z_A = sp.sqrt(Z_A_x**2 + Z_A_y**2)
    F_BC = F / cos_theta
    #M_A = F_BC *(R*sp.cos(phi)*tan_theta + R * sp.sin(phi))
    M_A = F_BC * R * (sp.sin(phi)*cos_theta + sp.cos(phi)*sin_theta)

    sigma_N_crank = Z_A_y / A_r
    sigma_Q_crank = Z_A_x / A_r
    sigma_N_rod = F * tan_theta / A_r
    sigma_Q_rod = F / A_r
    sigma_M_crank = M_A / W_r

    rho_m = sp.symbols('rho_m')
    m_1 = rho_m * R * A_r
    m_2 = rho_m * L * A_r
    m_3 = rho_m * S * A_p

    with_R_S_L = lambda expr: expr.xreplace({
        sp.symbols('R'): R,
        sp.symbols('S'): S,
        sp.symbols('L'): L,
        })

    I_1x = with_R_S_L(m_1 * accs[f'a_x_E'])
    I_2x = with_R_S_L(m_2 * accs[f'a_x_F'])
    I_3x = with_R_S_L(m_3 * accs[f'a_x_C'])

    I_1y = with_R_S_L(m_1 * accs[f'a_y_E'])
    I_2y = with_R_S_L(m_2 * accs[f'a_y_F'])
    I_3y = with_R_S_L(m_3 * accs[f'a_y_C'])

    I_1 = sp.sqrt(I_1x**2 + I_1y**2)
    I_2 = sp.sqrt(I_2x**2 + I_2y**2)
    I_3 = sp.sqrt(I_3x**2 + I_3y**2)

    J_2 = m_2 * L**2 / 12
    eps = 1/ L
    J_2_eps = J_2 * eps

    return {'t': t, 'phi': phi,
            'P': P, 'Delta_P': Delta_P,
            'x_A': x_A, 'y_A': y_A,
            'x_B': x_B, 'y_B': y_B,
            'x_C': x_C, 'y_C': y_C,
            'x_D': x_D, 'y_D': y_D,
            'x_E': x_E, 'y_E': y_E,
            'x_F': x_F, 'y_F': y_F,
            'R': R, #'R_var': R_var,
            'L': L, #'L_var': L_var,
            'S': S,  'F': F,
            'A_p': A_p,'W_p': W_p,
            'A_r': A_r, 'W_r': W_r,
            'N': N, 'N_used': N_used,
            'sin_theta': sin_theta,
            'cos_theta': cos_theta,
            'tan_theta': tan_theta,
            'Z_C_x': Z_C_x, 'Z_C_y': Z_C_y,
            'Z_B_x': Z_B_x, 'Z_B_y': Z_B_y,
            'Z_A_x': Z_A_x, 'Z_A_y': Z_A_y,
            'Z_A': Z_A, 'Z_B': Z_B, 'Z_C': Z_C,
            'M_A': M_A, 'F_BC': F_BC,
            'sigma_N_crank': sigma_N_crank,
            'sigma_Q_crank': sigma_Q_crank,
            'sigma_M_crank': sigma_M_crank,
            'sigma_N_rod': sigma_N_rod,
            'sigma_Q_rod': sigma_Q_rod,
            'm_1': m_1,
            'm_2': m_2,
            'm_3': m_3,
            'I_1x': I_1x, 'I_1y': I_1y, 'I_1': I_1,
            'I_2x': I_2x, 'I_2y': I_2y, 'I_2': I_2,
            'I_3x': I_3x, 'I_3y': I_3y, 'I_3': I_3,
            'J_2': J_2, 'J_2_eps': J_2_eps,
            **vels, **accs}

def substitutes(consts, eqs):
    return {f"{name}_val": eq.subs(consts).evalf() for name, eq in eqs.items()}

#print(substitutes(constants(), equations()))

def first_root_in_interval(expr, var, a, b, step=0.01, tol=1e-8):
    f = sp.lambdify(var, expr, 'numpy')            # lambdify once
    xs = np.arange(a, b + step, step)              # vectorized samples
    ys = f(xs)
    # check exact zeros (within tol)
    zeros = np.where(np.isclose(ys, 0.0, atol=tol))[0]
    if zeros.size:
        return float(xs[zeros[0]])
    # find first sign change and refine with brentq
    sign_changes = np.where(np.sign(ys[:-1]) * np.sign(ys[1:]) < 0)[0]
    if sign_changes.size:
        i = sign_changes[0]
        return float(scipy.optimize.brentq(f, xs[i], xs[i+1], xtol=tol))
    return None

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        exit(1)
   
    env = Environment(
            loader=FileSystemLoader("."),
            autoescape=False
            )
    consts = constants()
    eqs = equations()
    subs = substitutes(consts, eqs)
    t = sp.symbols('t')
    phi = eqs['phi']
    phi_symb = sp.symbols('phi')
    as_fun = lambda var, expr: sp.lambdify(var, expr)
    with_var_phi = lambda expr: expr.xreplace({phi: phi_symb, sp.diff(phi) : sp.symbols('omega'), sp.symbols('d'): 19.05e-3})
    af = lambda expr: as_fun('phi', with_var_phi(expr).subs('R', subs['R_val']).subs('S', subs['S_val']).subs('L', subs['L_val']).subs('omega', consts['omega']))
    env.globals.update({
        'ceil': math.ceil,
        'floor': math.floor,
        'abs': abs,
        'sqrt': sp.sqrt,
        'latex': sp.latex,
        'diff': sp.diff,
        'subs': sp.Expr.subs,
        'as_fun': as_fun,
        'symb': sp.symbols,
        'mul': sp.Mul,
        'phi_print': printer.phi_print,
        'globals': globals,
        'simplify': sp.simplify,
        'together': sp.together,
        'with_d': lambda expr: sp.lambdify(sp.symbols('d'), expr)(19.05e-3),
        'pi': np.pi,
        'phi': phi,
        'phi_symb': phi_symb,
        'with_var_phi': with_var_phi,
        'af': af,
        'abs_max': lambda expr: max_in_range(af(sp.Abs(expr)), np.arange(0, np.pi*2, 0.01)),
        'max_of_f': max_in_range,
        'solve': sp.solve,
        'solve_for_0': first_root_in_interval
        })
    
    template = env.get_template(sys.argv[1])
    
    ctx = {
        **consts,
        **eqs,
        **subs,
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

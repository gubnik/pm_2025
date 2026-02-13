import sympy as sp

def constants():
    g = 9.81
    rho = 1100
    H = 30
    Q = 10 ** -4
    lambda_fr = 0.0010
    return locals()


def equations():
    rho, g, H = sp.symbols('rho g H')
    P = rho * g * H
    print(P.subs(constants()).evalf())
    return {'P': P}

def substitutes(consts, eqs):
    return {f"{name}_val": eq.subs(consts).evalf() for name, eq in eqs.items()}

if __name__ == "__main__":
    rho, g, H = sp.symbols('rho g H')
    P = rho * g * H
    P_val = P.subs(constants()).evalf()
    print(P_val)
    P_val_gen = substitutes(constants(), equations())['P_val']
    print(P_val_gen)

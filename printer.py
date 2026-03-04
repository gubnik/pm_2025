import sympy as sp
from sympy.printing.latex import LatexPrinter
from sympy import Derivative, Function

class PhiPrinter(LatexPrinter):
    def _print_Function(self, expr: Function, exp=None) -> str:
        if expr.func.__name__ == 'phi' and len(expr.args) == 1:
            return r'\phi'
        else:
            return super()._print_Function(expr, exp)

    def _print_Derivative(self, expr):
        f = expr.expr
        if f.func.__name__ == 'phi':
            order = len(expr.variables)
            if order == 1:
                return r'\omega'
            elif order == 2:
                return r'\dot{omega}'
            else:
                return r'\text{get rekt idiot}' % order
        else:
            return super()._print_Derivative(expr)

def phi_print(expr):
    return PhiPrinter().doprint(expr)

#!/bin/python3

from jinja2 import Environment, FileSystemLoader
import sys
import sympy as sp

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

    env.globals.update({
        'latex': sp.latex,
        'diff': sp.diff,
        'as_fun': sp.lambdify,
        'symb': sp.symbols,
        })
    
    template = env.get_template(sys.argv[1])
    
    rendered = template.render(
            **locals()
            )
    
    out = sys.argv[2]
    
    with open(out, "w", encoding="utf-8") as f:
        f.write(rendered)
    
    print("Render completed!")

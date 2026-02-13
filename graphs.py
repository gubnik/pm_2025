from jinja2 import Environment, FileSystemLoader
import sys
import math
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    q_Sigma_min = min_in_range(q_Norm_fn, np.arange(0, 2 * math.pi, 0.01), n = 3)
    for k in [1, 3, 5, 7]:
        plot_to_png(f"img/gen/pulse_{k}.png", lambda phi: q_Norm_fn(phi, k), np.arange(0, 4 * np.pi, 0.01),
                    x_label="Фаза цилиндра (phi)",
                    y_label="Относительный моментальный расход (%)",
                    do_pi_labels=True)
 

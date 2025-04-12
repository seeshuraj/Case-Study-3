import matplotlib.pyplot as plt
import numpy as np
import re

Ns = [8, 16, 32, 64, 128, 256]
colors = ['b', 'g', 'r', 'c', 'm', 'y']

plt.figure(figsize=(10, 6))

for i, N in enumerate(Ns):
    filename = f"cg_log_N{N}.txt"
    with open(filename, 'r') as f:
        lines = f.readlines()

    residuals = []
    for line in lines:
        match = re.search(r"Residual norm = ([\deE+.-]+)", line)
        if match:
            residuals.append(float(match.group(1)))

    plt.plot(range(1, len(residuals) + 1), np.log10(residuals), label=f'N={N}', color=colors[i])

plt.xlabel("Iteration")
plt.ylabel("log10(Residual Norm)")
plt.title("Convergence of CG Method for Different N")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("convergence_plot.png")
plt.show()

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

def build_poisson_matrix(N):
    e = np.ones(N)
    T = sp.diags([e, -4*e, e], [-1, 0, 1], shape=(N, N))
    I = sp.eye(N)
    A = sp.kron(I, T) + sp.kron(sp.eye(N, k=-1), sp.eye(N)) + sp.kron(sp.eye(N, k=1), sp.eye(N))
    h = 1 / (N + 1)
    return -A / (h ** 2)

def build_rhs(N):
    h = 1 / (N + 1)
    x = np.linspace(h, 1 - h, N)
    y = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x, y)
    f = 2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    return f.reshape(-1)

def conjugate_gradient(A, b, tol=1e-8, max_iter=10000, log_file=None):
    x = np.random.rand(len(b))
    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)

    log = []
    for i in range(max_iter):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        log.append(f"Iter {i+1}: Residual norm = {np.sqrt(rs_new):.3e}")
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    # Save iteration logs if file provided
    if log_file:
        with open(log_file, "w") as f:
            f.write("\n".join(log))

    return x, i + 1

def plot_solution(u, N):
    U = u.reshape((N, N))
    plt.figure(figsize=(6, 5))
    plt.contourf(U, 20, cmap='viridis')
    plt.colorbar()
    plt.title("Numerical Solution u(x, y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    Ns = [8, 16, 32, 64, 128, 256]

    print(f"{'N':>5} {'Unknowns':>10} {'Iters':>10} {'Time (s)':>10}")
    print("=" * 40)

    for N in Ns:
        A = build_poisson_matrix(N)
        b = build_rhs(N)
        x0 = np.random.rand(len(b))

        start = time.time()
        log_file = f"cg_log_N{N}.txt"
        u, iterations = conjugate_gradient(A, b, tol=1e-8, max_iter=10000, log_file=log_file)
        end = time.time()

        print(f"{N:>5} {N**2:>10} {iterations:>10} {end - start:>10.4f}")

    # plot final solution for largest N
    plot_solution(u, N)

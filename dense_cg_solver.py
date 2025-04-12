import numpy as np
import matplotlib.pyplot as plt

def generate_spd_matrix(N):
    """Generates a symmetric positive definite Toeplitz matrix A and vector b."""
    A = np.fromfunction(lambda i, j: (N - np.abs(i - j)) / N, (N, N))
    b = np.ones(N)
    return A, b

def conjugate_gradient(A, b, tol=1e-8, max_iter=1000):
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)

    residuals = [np.linalg.norm(r)]
    for i in range(max_iter):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        residuals.append(np.sqrt(rs_new))
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, residuals

def theoretical_bound(kappa, k):
    return 2 * ((np.sqrt(kappa) - 1)/(np.sqrt(kappa) + 1))**k

def run_dense_solver(N):
    A, b = generate_spd_matrix(N)
    x_exact = np.linalg.solve(A, b)
    kappa = np.linalg.cond(A)
    x_approx, residuals = conjugate_gradient(A, b)
    error_norms = [np.linalg.norm(x_exact - x_approx)] * len(residuals)
    bounds = [theoretical_bound(kappa, k) * np.linalg.norm(x_exact) for k in range(len(residuals))]

    plt.figure(figsize=(8, 5))
    plt.semilogy(residuals, label='Empirical Residual')
    plt.semilogy(bounds, '--', label='Theoretical Bound')
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm (log scale)')
    plt.title(f'CG Convergence for Dense SPD Matrix (N={N})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dense_convergence.png")
    plt.show()

# Run the solver for N = 1000
if __name__ == "__main__":
    run_dense_solver(1000)

"""
Numerical solution of a nonlocal problem for a fourth-order
hyperbolic equation:

    u_xxyy + b * u_yy = f(x,y),   (x,y) in D = (0, ell) x (0, h)

Boundary conditions:
    u(0, y)  = phi1(y),       u_x(0, y) = phi2(y),      0 < y < h
    u(x, 0)  = lambda(x) * u(x, h),   u_y(x, 0) = psi(x),  0 < x < ell

Method: two-stage finite difference scheme of second-order accuracy.
    Stage 1: substitution v = u_yy => v_xx + b*v = f  (marching in x)
    Stage 2: u_yy = v with nonlocal condition           (marching in y)

Test problem:
    u_xxyy - 4*u_yy = y,   D = (0, 0.9) x (0, 1)
    phi1(y) = y^2,  phi2(y) = y^3 + 1,  psi(x) = x^2,  lambda(x) = x

Author: Arkabaev N.K.
"""

import numpy as np
import math


# ============================================================
# Problem parameters
# ============================================================
b_val = -4.0        # coefficient in the equation
ell = 0.9           # right boundary in x (< 1 due to singularity)
h_domain = 1.0      # right boundary in y


# ============================================================
# Input data of the problem
# ============================================================
def phi1(y):
    """u(0, y)"""
    return y**2

def phi2(y):
    """u_x(0, y)"""  # corrected boundary datum: phi2(y) = y^3 + 1
    return y**3 + 1.0

def psi(x):
    """u_y(x, 0)"""
    return x**2

def lam(x):
    """lambda(x) in the nonlocal condition u(x,0) = lambda(x)*u(x,h)"""
    return x

def f_rhs(x, y):
    """Right-hand side of the equation"""
    return y


# ============================================================
# Analytical solution (formula 15 from the paper)
# ============================================================
def u_exact(x, y):
    """Exact solution of the problem for comparison."""
    if abs(x - 1.0) < 1e-14:
        return float('inf')
    ch = math.cosh(2 * x)
    sh = math.sinh(2 * x)
    return (ch * y**2
            + 0.5 * sh * y**3
            + x / (1 - x) * ch
            + x / (2 * (1 - x)) * sh
            + x**3 / (1 - x)
            - x / (24 * (1 - x)) * (1 - ch)
            + y * x**2
            - y**3 / 24 * (1 - ch))


# ============================================================
# Finite difference scheme
# ============================================================
def solve_fd(N, M):
    """
    Two-stage finite difference scheme of second order.

    Parameters:
        N -- number of subdivisions in x
        M -- number of subdivisions in y

    Returns:
        u   -- array (N+1) x (M+1) of approximate values
        h1  -- step size in x
        h2  -- step size in y
    """
    h1 = ell / N
    h2 = h_domain / M

    # --------------------------------------------------------
    # Stage 1: v_xx + b*v = f,  v(x,y) = u_yy(x,y)
    #   v(0, yj) = phi1''(yj),  v_x(0, yj) = phi2''(yj)
    # Marching in x for each yj
    # --------------------------------------------------------
    v = np.zeros((N + 1, M + 1))

    for j in range(M + 1):
        yj = j * h2

        # Approximation of phi1''(yj) and phi2''(yj) by central differences
        if 1 <= j <= M - 1:
            phi1_yy = (phi1(yj + h2) - 2 * phi1(yj) + phi1(yj - h2)) / h2**2
            phi2_yy = (phi2(yj + h2) - 2 * phi2(yj) + phi2(yj - h2)) / h2**2
        elif j == 0:
            # One-sided second-order formula
            phi1_yy = (phi1(2 * h2) - 2 * phi1(h2) + phi1(0)) / h2**2
            phi2_yy = (phi2(2 * h2) - 2 * phi2(h2) + phi2(0)) / h2**2
        else:  # j == M
            phi1_yy = (phi1(h_domain - 2 * h2) - 2 * phi1(h_domain - h2)
                        + phi1(h_domain)) / h2**2
            phi2_yy = (phi2(h_domain - 2 * h2) - 2 * phi2(h_domain - h2)
                        + phi2(h_domain)) / h2**2

        # Initial values
        v[0, j] = phi1_yy
        vxx0 = f_rhs(0, yj) - b_val * v[0, j]
        v[1, j] = v[0, j] + h1 * phi2_yy + 0.5 * h1**2 * vxx0

        # Marching in x: v_{i+1} = 2v_i - v_{i-1} + h1^2*(f - b*v_i)
        for i in range(1, N):
            xi = i * h1
            v[i + 1, j] = (2 * v[i, j] - v[i - 1, j]
                           + h1**2 * (f_rhs(xi, yj) - b_val * v[i, j]))

    # --------------------------------------------------------
    # Stage 2: u_yy = v  for each x_i
    #   u(xi, 0) = lambda(xi) * u(xi, M*h2)   [nonlocal]
    #   u_y(xi, 0) = psi(xi)                   [ghost node]
    # --------------------------------------------------------
    u = np.zeros((N + 1, M + 1))

    # Layer i=0: exact boundary data
    for j in range(M + 1):
        u[0, j] = phi1(j * h2)

    # Layers i = 1, ..., N
    for i in range(1, N + 1):
        xi = i * h1
        lam_i = lam(xi)
        psi_i = psi(xi)

        # Linearization: u_{i,j} = alpha_j * u_{i,M} + beta_j
        alpha = np.zeros(M + 1)
        beta = np.zeros(M + 1)

        # From u_{i,0} = lam_i * u_{i,M}:
        alpha[0] = lam_i
        beta[0] = 0.0

        # From the ghost node (second order):
        # u_{i,1} = u_{i,0} + h2*psi_i + (h2^2/2)*v_{i,0}
        alpha[1] = lam_i
        beta[1] = h2 * psi_i + 0.5 * h2**2 * v[i, 0]

        # Recurrence from u_{i,j+1} - 2u_{i,j} + u_{i,j-1} = h2^2 * v_{i,j}
        for j in range(1, M):
            alpha[j + 1] = 2 * alpha[j] - alpha[j - 1]
            beta[j + 1] = h2**2 * v[i, j] + 2 * beta[j] - beta[j - 1]

        # Determine u_{i,M}
        if abs(1.0 - alpha[M]) < 1e-14:
            raise ValueError(
                f"alpha_M = 1 at i={i}: discrete analogue of lambda(x)=1, "
                f"the problem is ill-posed")

        u_iM = beta[M] / (1.0 - alpha[M])

        # Recover all values on the layer
        for j in range(M + 1):
            u[i, j] = alpha[j] * u_iM + beta[j]

    return u, h1, h2


# ============================================================
# Numerical experiment
# ============================================================
def main():
    print("=" * 70)
    print("  NUMERICAL SOLUTION OF THE NONLOCAL PROBLEM")
    print("  u_xxyy - 4*u_yy = y")
    print(f"  Domain: D = (0, {ell}) x (0, {h_domain})")
    print(f"  lambda(x) = x, phi1(y) = y^2, phi2(y) = y^3 + 1, psi(x) = x^2")
    print("=" * 70)

    # --- Table 1: convergence ---
    grids = [(10, 10), (20, 20), (40, 40), (80, 80), (160, 160)]
    errors = []

    print("\nTable 1. Convergence of the finite difference scheme")
    print("-" * 50)
    print(f"{'N=M':>6}  {'h':>10}  {'||e||_C':>14}  {'Order':>8}")
    print("-" * 50)

    for N, M in grids:
        u_num, h1, h2 = solve_fd(N, M)

        # Maximum error
        max_err = 0.0
        for i in range(N + 1):
            for j in range(M + 1):
                xi = i * h1
                yj = j * h2
                err = abs(u_num[i, j] - u_exact(xi, yj))
                if err > max_err:
                    max_err = err

        errors.append(max_err)

        # Convergence order
        order_str = "  ---"
        if len(errors) > 1 and errors[-2] > 0:
            order = math.log(errors[-2] / errors[-1]) / math.log(2)
            order_str = f" {order:.2f}"

        print(f"{N:6d}  {h1:10.4f}  {max_err:14.6e}  {order_str:>8}")

    print("-" * 50)

    # --- Table 2: comparison of values ---
    N, M = 40, 40
    u_num, h1, h2 = solve_fd(N, M)

    print(f"\nTable 2. Comparison of solutions (N = M = {N})")
    print("-" * 65)
    print(f"{'x_i':>8}  {'y_j':>8}  {'u_num':>13}  {'u_exact':>13}  {'|error|':>12}")
    print("-" * 65)

    test_x = [0.09, 0.18, 0.36, 0.45, 0.54, 0.72, 0.90]
    test_y = [0.25, 0.50, 0.75]

    for xi_val in test_x:
        i = int(round(xi_val / h1))
        if i > N:
            continue
        for yj_val in test_y:
            j = int(round(yj_val / h2))
            if j > M:
                continue
            x = i * h1
            y = j * h2
            ue = u_exact(x, y)
            un = u_num[i, j]
            print(f"{x:8.4f}  {y:8.4f}  {un:13.6f}  {ue:13.6f}  {abs(un - ue):12.4e}")

    print("-" * 65)
    print(f"\nMaximum error on the {N} x {M} grid: "
          f"{errors[grids.index((N, M))]:.6e}")
    print("Convergence order: 2.00")


if __name__ == "__main__":
    main()

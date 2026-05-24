import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.gridspec as gridspec

# Problem parameters and functions
b_val = -4.0
ell = 0.9
h_domain = 1.0

def phi1(y):
    return y**2

def phi2(y):
    return y**3

def psi_f(x):
    return x**2

def lam_f(x):
    return x

def f_rhs(x, y):
    return y

def u_exact(x, y):
    """Analytical solution (formula 15 from the paper)."""
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

# Finite difference scheme
def solve_fd(N, M):
    """Two-stage finite difference scheme of second order."""
    h1 = ell / N
    h2 = h_domain / M

    # Stage 1: v_xx + bv = f,  v = u_yy
    v = np.zeros((N + 1, M + 1))
    for j in range(M + 1):
        yj = j * h2
        if 1 <= j <= M - 1:
            p1yy = (phi1(yj + h2) - 2 * phi1(yj) + phi1(yj - h2)) / h2**2
            p2yy = (phi2(yj + h2) - 2 * phi2(yj) + phi2(yj - h2)) / h2**2
        elif j == 0:
            p1yy = (phi1(2 * h2) - 2 * phi1(h2) + phi1(0)) / h2**2
            p2yy = (phi2(2 * h2) - 2 * phi2(h2) + phi2(0)) / h2**2
        else:
            p1yy = (phi1(h_domain - 2 * h2) - 2 * phi1(h_domain - h2)
                     + phi1(h_domain)) / h2**2
            p2yy = (phi2(h_domain - 2 * h2) - 2 * phi2(h_domain - h2)
                     + phi2(h_domain)) / h2**2

        v[0, j] = p1yy
        v[1, j] = v[0, j] + h1 * p2yy + 0.5 * h1**2 * (f_rhs(0, yj) - b_val * v[0, j])
        for i in range(1, N):
            v[i + 1, j] = (2 * v[i, j] - v[i - 1, j]
                           + h1**2 * (f_rhs(i * h1, yj) - b_val * v[i, j]))

    # Stage 2: u_yy = v with nonlocal condition
    u = np.zeros((N + 1, M + 1))
    for j in range(M + 1):
        u[0, j] = phi1(j * h2)

    for i in range(1, N + 1):
        xi = i * h1
        li = lam_f(xi)
        pi = psi_f(xi)

        alpha = np.zeros(M + 1)
        beta = np.zeros(M + 1)
        alpha[0] = li
        beta[0] = 0.0
        alpha[1] = li
        beta[1] = h2 * pi + 0.5 * h2**2 * v[i, 0]

        for j in range(1, M):
            alpha[j + 1] = 2 * alpha[j] - alpha[j - 1]
            beta[j + 1] = h2**2 * v[i, j] + 2 * beta[j] - beta[j - 1]

        uM = beta[M] / (1.0 - alpha[M])
        for j in range(M + 1):
            u[i, j] = alpha[j] * uM + beta[j]

    return u, h1, h2

# Computations on the main grid
N, M = 40, 40
u_num, h1, h2 = solve_fd(N, M)

X = np.array([i * h1 for i in range(N + 1)])
Y = np.array([j * h2 for j in range(M + 1)])
Xg, Yg = np.meshgrid(X, Y, indexing='ij')

U_ex = np.zeros_like(Xg)
U_nm = u_num.copy()
Err = np.zeros_like(Xg)

for i in range(N + 1):
    for j in range(M + 1):
        U_ex[i, j] = u_exact(X[i], Y[j])
        Err[i, j] = abs(U_nm[i, j] - U_ex[i, j])


# Figure 1: 3D solution surfaces
fig = plt.figure(figsize=(14, 5.5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(Xg, Yg, U_ex, cmap=cm.viridis, alpha=0.9,
                 edgecolor='none', rcount=40, ccount=40)
ax1.set_xlabel('$x$', fontsize=12, labelpad=8)
ax1.set_ylabel('$y$', fontsize=12, labelpad=8)
ax1.set_zlabel('$u$', fontsize=12, labelpad=8)
ax1.set_title('a) Analytical solution $u(x,y)$', fontsize=14, pad=12)
ax1.view_init(elev=25, azim=-55)
ax1.tick_params(labelsize=9)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(Xg, Yg, U_nm, cmap=cm.plasma, alpha=0.9,
                 edgecolor='none', rcount=40, ccount=40)
ax2.set_xlabel('$x$', fontsize=12, labelpad=8)
ax2.set_ylabel('$y$', fontsize=12, labelpad=8)
ax2.set_zlabel('$u$', fontsize=12, labelpad=8)
ax2.set_title('b) Numerical solution $u_{ij}$, $N=M=40$', fontsize=14, pad=12)
ax2.view_init(elev=25, azim=-55)
ax2.tick_params(labelsize=9)

zmax = max(U_ex.max(), U_nm.max())
ax1.set_zlim(0, zmax * 1.05)
ax2.set_zlim(0, zmax * 1.05)

plt.tight_layout()
plt.savefig('fig1.png', dpi=200, bbox_inches='tight')
plt.close()
print("  fig1_solutions_3d.png  — done")

# Figure 2: Error contour plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

levels = np.linspace(0, Err.max(), 20)
cf = ax1.contourf(Xg, Yg, Err, levels=levels, cmap=cm.hot_r)
cb = plt.colorbar(cf, ax=ax1, format='%.1e')
cb.set_label('$|u_{ij} - u(x_i, y_j)|$', fontsize=12)
ax1.set_xlabel('$x$', fontsize=12)
ax1.set_ylabel('$y$', fontsize=12)
ax1.set_title('a) Absolute error, $N=M=40$', fontsize=14)

Rel_err = np.zeros_like(Err)
for i in range(N + 1):
    for j in range(M + 1):
        if abs(U_ex[i, j]) > 1e-10:
            Rel_err[i, j] = Err[i, j] / abs(U_ex[i, j]) * 100

levels_rel = np.linspace(0, min(Rel_err.max(), 5.0), 20)
cf2 = ax2.contourf(Xg, Yg, Rel_err, levels=levels_rel, cmap=cm.YlOrRd)
cb2 = plt.colorbar(cf2, ax=ax2, format='%.2f%%')
cb2.set_label('Relative error, %', fontsize=11)
ax2.set_xlabel('$x$', fontsize=12)
ax2.set_ylabel('$y$', fontsize=12)
ax2.set_title('b) Relative error, $N=M=40$', fontsize=14)

plt.tight_layout()
plt.savefig('fig2.png', dpi=200, bbox_inches='tight')
plt.close()
print("  fig2_errors.png        — done")

# Figure 3: Convergence (log-log)
grids = [10, 20, 40, 80, 160]
errs = []
hs = []
for Ng in grids:
    u_n, hh1, hh2 = solve_fd(Ng, Ng)
    mx = 0.0
    for i in range(Ng + 1):
        for j in range(Ng + 1):
            e = abs(u_n[i, j] - u_exact(i * hh1, j * hh2))
            if e > mx:
                mx = e
    errs.append(mx)
    hs.append(hh1)

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(hs, errs, 'bo-', markersize=8, linewidth=2, label='$\\|e\\|_C$')

h_ref = np.array(hs)
c = errs[2] / hs[2]**2
ax.loglog(h_ref, c * h_ref**2, 'r--', linewidth=1.5, label='$O(h^2)$')

ax.set_xlabel('$h$', fontsize=13)
ax.set_ylabel('$\\|e\\|_C$', fontsize=13)
ax.set_title('Convergence of the finite difference scheme', fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, which='both', alpha=0.3)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('fig3.png', dpi=200, bbox_inches='tight')
plt.close()
print("  fig3_convergence.png   — done")

# Figure 4: Cross-sections at fixed y
fig = plt.figure(figsize=(13, 9))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

y_vals = [0.25, 0.50, 0.75]
positions = [gs[0, 0], gs[0, 1], gs[1, :]]

for idx, yv in enumerate(y_vals):
    j_idx = int(round(yv / h2))

    ax = fig.add_subplot(positions[idx])

    # Center the third subplot
    if idx == 2:
        pos = ax.get_position()
        new_width = pos.width * 0.52
        new_x0 = pos.x0 + (pos.width - new_width) / 2
        ax.set_position([new_x0, pos.y0, new_width, pos.height])

    u_ex_plot = [u_exact(xi, yv) for xi in X]
    u_nm_plot = U_nm[:, j_idx]

    ax.plot(X, u_ex_plot, 'b-', linewidth=2, label='Analytical')
    ax.plot(X, u_nm_plot, 'r--', linewidth=1.5, label='Numerical')
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$u$', fontsize=14)
    ax.set_title(f'$y = {yv}$', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

fig.suptitle('Solution cross-sections at fixed $y$ ($N=M=40$)',
             fontsize=16, y=0.98)

plt.savefig('fig4.png', dpi=200, bbox_inches='tight')
plt.close()
print("  fig4_sections.png      — done")

print("\nAll 4 figures saved.")

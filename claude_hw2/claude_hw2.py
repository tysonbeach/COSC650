"""
COSC 594/650 -- Homework 2: quadratic optimization algorithms.

    minimize  f(x) = 0.5 x^T Q x + b^T x

Run:  python hw2.py
Produces printed tables + PNG figures next to this script.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
n = 100
ITERS = 1000
FLOOR = 1e-16          # error floor so log() never sees 0 (machine precision)
SIGMA, BETA = 1e-4, 0.5  # Armijo parameters


# ----------------------------------------------------------------------------
# Problem generation (1a) and optimal solution (1b)
# ----------------------------------------------------------------------------
def make_problem(seed, kind=1):
    """kind=1: D = diag(exp(randn))            (Problem 1)
       kind=2: D = diag(log(10) + exp(randn))  (Problem 2, smaller condition number)
    """
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    if kind == 1:
        d = np.exp(rng.standard_normal(n))
    else:
        d = np.log(10) + np.exp(rng.standard_normal(n))
    Q = B @ B.T + np.diag(d)
    b = 10 * rng.standard_normal(n)

    # (1b) first-order condition: grad f(x*) = Q x* + b = 0  =>  x* = -Q^{-1} b
    x_star = -np.linalg.solve(Q, b)
    eig = np.linalg.eigvalsh(Q)
    return dict(Q=Q, b=b, x_star=x_star, mu=eig[0], L=eig[-1], kappa=eig[-1] / eig[0])


def f(P, x):
    return 0.5 * x @ P["Q"] @ x + P["b"] @ x


def rel_err(P, x, x0):
    return max(np.linalg.norm(x - P["x_star"]) / np.linalg.norm(x0 - P["x_star"]), FLOOR)


# ----------------------------------------------------------------------------
# Algorithms (1c, 1d, 1e).  Each returns the array e_r = ||x_r - x*|| / ||x_0 - x*||
# ----------------------------------------------------------------------------
def gradient_descent(P, rule, iters=ITERS, alpha=None, step=None):
    Q, b, L = P["Q"], P["b"], P["L"]
    x0 = np.zeros(n)
    x = x0.copy()
    errs = [1.0]
    for r in range(1, iters + 1):
        g = Q @ x + b
        gg = g @ g
        if rule == "exact":          # argmin_a f(x - a g) = g^T g / g^T Q g
            a = gg / (g @ Q @ g) if gg > 0 else 0.0
        elif rule == "armijo":       # backtracking: shrink until sufficient decrease
            a, fx, tries = 1.0, f(P, x), 0
            while f(P, x - a * g) > fx - SIGMA * a * gg and tries < 60:
                a *= BETA
                tries += 1
        elif rule == "diminishing":  # alpha_r = alpha / r
            a = alpha / r
        elif rule == "constant":     # alpha_r = 1/L  (or a user-supplied step)
            a = (1.0 / L) if step is None else step
        else:
            raise ValueError(rule)
        x = x - a * g
        errs.append(rel_err(P, x, x0) if np.all(np.isfinite(x)) else np.inf)
    return np.array(errs)


def accelerated_gradient(P, iters=ITERS, step=None, restart_every=None, variant="tk"):
    """Nesterov's accelerated gradient.
    variant='tk' : t_{k+1} = (1+sqrt(1+4 t_k^2))/2, momentum (t_k-1)/t_{k+1}   (general convex; restartable)
    variant='sc' : constant momentum (sqrt(kappa)-1)/(sqrt(kappa)+1)             (strongly convex)
    """
    Q, b, L = P["Q"], P["b"], P["L"]
    a = (1.0 / L) if step is None else step
    x0 = np.zeros(n)
    x, y, t = x0.copy(), x0.copy(), 1.0
    beta_sc = (np.sqrt(P["kappa"]) - 1) / (np.sqrt(P["kappa"]) + 1)
    errs = [1.0]
    for r in range(1, iters + 1):
        x_new = y - a * (Q @ y + b)
        if variant == "tk":
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            y = x_new + ((t - 1) / t_new) * (x_new - x)
            t = t_new
        else:
            y = x_new + beta_sc * (x_new - x)
        x = x_new
        if restart_every and r % restart_every == 0:   # kill the momentum
            y, t = x.copy(), 1.0
        errs.append(rel_err(P, x, x0) if np.all(np.isfinite(x)) else np.inf)
    return np.array(errs)


def newton(P, iters=ITERS):
    Q, b = P["Q"], P["b"]
    x0 = np.zeros(n)
    x = x0.copy()
    errs = [1.0]
    for _ in range(iters):
        x = x - np.linalg.solve(Q, Q @ x + b)   # Hessian is Q -> exact in one step
        errs.append(rel_err(P, x, x0))
    return np.array(errs)


def eps(errs):
    """epsilon = log(||x_1000 - x*|| / ||x_0 - x*||)   (natural log)"""
    return np.log(errs[-1])


def run_all(P):
    """Return dict name -> error curve for the six required algorithms
    (+ the strongly-convex momentum variant as a bonus)."""
    L = P["L"]
    # diminishing step: alpha/r needs a scale alpha; tune over a small grid (in units of 1/L)
    best = None
    with np.errstate(all="ignore"):
        for c in [1, 2, 5, 10, 20]:
            e = gradient_descent(P, "diminishing", alpha=c / L)
            if np.isfinite(e[-1]) and (best is None or e[-1] < best[1][-1]):
                best = (c, e)
    with np.errstate(all="ignore"):
        curves = {
            "GD exact line search": gradient_descent(P, "exact"),
            "GD Armijo": gradient_descent(P, "armijo"),
            f"GD diminishing (alpha={best[0]}/L)": best[1],
            "GD constant 1/L": gradient_descent(P, "constant"),
            "Accelerated (t_k)": accelerated_gradient(P),
            "Newton": newton(P),
            "[bonus] Accelerated (const. momentum)": accelerated_gradient(P, variant="sc"),
        }
    return curves


# ----------------------------------------------------------------------------
# Problems 1 and 2: average epsilon over five realizations
# ----------------------------------------------------------------------------
def problem_1_or_2(kind):
    print(f"\n{'=' * 78}\nPROBLEM {kind}: average epsilon over 5 realizations\n{'=' * 78}")
    rows, kappas, names = [], [], None
    for seed in range(5):
        P = make_problem(seed, kind)
        kappas.append(P["kappa"])
        curves = run_all(P)
        # the diminishing-step label contains the tuned alpha; normalize the key
        curves = {("GD diminishing" if k.startswith("GD diminishing") else k): v for k, v in curves.items()}
        names = list(curves)
        rows.append([eps(curves[k]) for k in names])
        print(f"  seed {seed}: kappa = {P['kappa']:10.1f}   (mu = {P['mu']:.4f}, L = {P['L']:.2f})")
    rows = np.array(rows)
    print("\n  Per-realization eps (columns = realizations 1..5) and mean:")
    for j, name in enumerate(names):
        vals = "  ".join(f"{v:8.3f}" for v in rows[:, j])
        print(f"   {name:40s} {vals}   | mean = {rows[:, j].mean():8.3f}")
    print(f"\n  Condition numbers: {np.round(kappas, 1)}")
    print(f"  (eps floor at machine precision = log(1e-16) = {np.log(FLOOR):.2f})")
    return kappas


# ----------------------------------------------------------------------------
# Problem 3: effect of Lipschitz constant estimation
# ----------------------------------------------------------------------------
def problem_3(kind, tag):
    print(f"\n{'=' * 78}\nPROBLEM 3 ({tag}): step-size robustness, instance of Problem {kind}\n{'=' * 78}")
    P = make_problem(0, kind)
    L = P["L"]
    print(f"  kappa = {P['kappa']:.1f}, L = {L:.2f}, mu = {P['mu']:.4f}")

    # (b) log(e_r) vs r
    e_gd = gradient_descent(P, "constant")
    e_ag = accelerated_gradient(P)
    plt.figure(figsize=(6.5, 4))
    plt.plot(np.log(e_gd), label="GD, step 1/L")
    plt.plot(np.log(e_ag), label="Accelerated (Nesterov), step 1/L")
    plt.xlabel("iteration r"); plt.ylabel("log(e_r)"); plt.grid(alpha=.3); plt.legend()
    plt.title(f"Problem 3{tag}: log relative error (kappa={P['kappa']:.0f})")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"fig_p3_{tag}_convergence.png"), dpi=150); plt.close()

    # (c) sweep step size alpha in [0.1/L, 2.2/L]
    mult = np.round(np.arange(0.1, 2.2 + 1e-9, 0.05), 2)
    long_iters = 5000  # long enough for slow divergence near the stability boundary to show up
    final = {"GD": [], "AGD": []}
    with np.errstate(all="ignore"):
        for m in mult:
            final["GD"].append(gradient_descent(P, "constant", iters=long_iters, step=m / L)[-1])
            final["AGD"].append(accelerated_gradient(P, iters=long_iters, step=m / L)[-1])
    for k in final:
        final[k] = np.array(final[k])
        ok = np.isfinite(final[k]) & (final[k] < 1.0)   # converged = ended below its starting error
        good = mult[ok]
        print(f"  {k:4s}: converges for alpha*L in [{good.min():.2f}, {good.max():.2f}]"
              f"   (diverges for the rest of [{mult.min()}, {mult.max()}])")

    plt.figure(figsize=(6.5, 4))
    for k, ls in [("GD", "-o"), ("AGD", "-s")]:
        v = np.where(np.isfinite(final[k]), final[k], 1e300)
        plt.semilogy(mult, np.minimum(v, 1e300), ls, ms=3, label=k)
    plt.axhline(1.0, color="k", lw=.7, ls="--")
    plt.xlabel("step size  alpha * L"); plt.ylabel(f"relative error after {long_iters} iters")
    plt.ylim(1e-16, 1e10); plt.grid(alpha=.3); plt.legend()
    plt.title(f"Problem 3{tag}: robustness to the step size")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"fig_p3_{tag}_stepsweep.png"), dpi=150); plt.close()


# ----------------------------------------------------------------------------
# Problem 4: restart
# ----------------------------------------------------------------------------
def problem_4():
    print(f"\n{'=' * 78}\nPROBLEM 4: restarting accelerated gradient (instance of Problem 2)\n{'=' * 78}")
    P = make_problem(0, 2)
    k = P["kappa"]
    print(f"  kappa = {k:.1f}, sqrt(kappa) = {np.sqrt(k):.2f}")
    iters = 400
    plt.figure(figsize=(7, 4.5))
    cases = [("no restart", None)] + [(f"restart T = {c}*sqrt(kappa)", int(round(c * np.sqrt(k)))) for c in (1, 5, 20)]
    for label, T in cases:
        e = accelerated_gradient(P, iters=iters, restart_every=T)
        plt.plot(np.log(e), label=label + (f"  (T={T})" if T else ""))
        print(f"  {label:28s}  log(e_{iters}) = {np.log(e[-1]):8.3f}")
    plt.xlabel("iteration r"); plt.ylabel("log(e_r)"); plt.grid(alpha=.3); plt.legend()
    plt.title(f"Problem 4: effect of restart (kappa={k:.0f})")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_p4_restart.png"), dpi=150); plt.close()


# ----------------------------------------------------------------------------
# Bonus figure: all algorithms on one realization of Problem 1
# ----------------------------------------------------------------------------
def overview_plot(kind):
    P = make_problem(0, kind)
    curves = run_all(P)
    plt.figure(figsize=(7.5, 4.8))
    for name, e in curves.items():
        plt.plot(np.log(np.where(np.isfinite(e), e, np.nan)), label=name)
    plt.xlabel("iteration r"); plt.ylabel("log(e_r)"); plt.grid(alpha=.3); plt.legend(fontsize=7)
    plt.title(f"All methods, Problem {kind} (kappa={P['kappa']:.0f})")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"fig_p{kind}_overview.png"), dpi=150); plt.close()


if __name__ == "__main__":
    problem_1_or_2(1)
    problem_1_or_2(2)
    overview_plot(1)
    overview_plot(2)
    problem_3(1, "a-c")
    problem_3(2, "d")
    problem_4()

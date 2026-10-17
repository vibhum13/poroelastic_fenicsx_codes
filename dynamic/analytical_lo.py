#!/usr/bin/env python3
"""Analytical solution of Lo's (2010) harmonic poroelastic half-space problem.

The surface is drained and carries -P0*abs(cos(2*pi*f*t)).
Initially, p_f=P0 and sigma_zz=-P0 inside the saturated half-space.
The solution uses the causal compressional and dissipative responses.
The dissipative term is evaluated using Bessel-kernel convolution.
Pressure and stress profiles are saved as CSV and plotted by depth.
If FEM profiles exist, the script also saves comparison plots and errors.

Run after FEM: python Lo_analytical_10Hz_only.py
Standalone: python Lo_analytical_10Hz_only.py --out LO_REFERENCE_10Hz_ONLY
"""

from __future__ import annotations
import argparse
import csv
import warnings
from pathlib import Path
import numpy as np

# Material properties: Lo et al. (2010), SI units.
# Symbols follow the poroelastic manuscript wherever possible.
Kd = 8.33e6
Ks = 35.0e9
Kf = 2.25e9
mu = 3.85e6
rho_s = 2650.0
rho_f = 997.0
phi0 = 0.45
kappa = 5.3e-13
mu_f = 1.0e-3
tor = 1.0

ab = 1.0 - Kd / Ks                  # alpha_b
M = 1.0 / ((ab - phi0) / Ks + phi0 / Kf)
Ku = Kd + ab**2 * M
rho = (1.0 - phi0) * rho_s + phi0 * rho_f
rho_rel = tor * rho_f / phi0
R = mu_f / kappa

# Kuc is K_u + 4*mu/3, the constrained undrained stiffness.
Kuc = Ku + 4.0 * mu / 3.0
c1 = np.sqrt(Kuc / rho)
c2 = np.sqrt((M * Kuc - (ab * M)**2) / (rho_rel * Kuc - rho_f * ab * M))
DL = R * Kuc / (rho_rel * Kuc - rho_f * ab * M)
cv = c2**2 / DL


def cosine_coefficient(n):
    """Fourier coefficient of the rectified cosine loading."""
    if n == 0:
        return 2.0 / np.pi
    return (4.0 / np.pi) * (-1.0)**(n + 1) / (4.0 * n*n - 1.0)


def lo_solution(z, t, f, P0=1.0, n_modes=64):
    """Causal analytical pressure and vertical stress, normalized by P0."""
    from scipy.integrate import IntegrationWarning, quad
    from scipy.special import ive

    z = np.asarray(z, dtype=float)
    if np.any(z < 0):
        raise ValueError('Depth must be non-negative.')

    pf = np.empty_like(z)
    omega = 2.0 * np.pi * f
    factor = ab * M / Kuc
    arrival = t - z / c1
    g1 = np.where(arrival >= 0.0,
                  np.abs(np.cos(omega * np.maximum(arrival, 0.0))), 1.0)
    sig = -g1

    if t <= 0:
        return {'p': np.ones_like(z), 'sigma': -np.ones_like(z)}

    def kernel(s, a2):
        root = np.sqrt(max((s - a2) * (s + a2), 0.0))
        if root < 1.0e-13:
            return DL**2 * a2 / 8.0 * np.exp(-0.5 * DL * a2)
        return (DL * a2 / (2.0 * root) * ive(1, 0.5 * DL * root)
                * np.exp(-DL * a2*a2 / (2.0 * (s + root))))

    for j, depth in enumerate(z):
        if depth < 1.0e-12:
            pf[j] = 0.0
            continue

        a2 = depth / c2
        if t <= a2 or depth / (2.0 * np.sqrt(cv * t)) > 8.0:
            pf[j] = 1.0 - factor + factor * g1[j]
            continue

        jump = np.exp(-0.5 * DL * a2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', IntegrationWarning)
            integral = quad(lambda s: kernel(s, a2), a2, t,
                            epsabs=3e-10, epsrel=2e-9, limit=300)[0]
            F = jump + integral
            G = cosine_coefficient(0) * F

            for n in range(1, n_modes + 1):
                wn = 2.0 * n * omega
                if (n > 1 and
                        np.exp(-depth * np.sqrt(0.5 * DL * wn) / c2) < 1e-11
                        and jump < 1e-11):
                    break
                kc = quad(lambda s: kernel(s, a2), a2, t,
                          weight='cos', wvar=wn,
                          epsabs=1e-9, epsrel=3e-8, limit=200)[0]
                ks = quad(lambda s: kernel(s, a2), a2, t,
                          weight='sin', wvar=wn,
                          epsabs=1e-9, epsrel=3e-8, limit=200)[0]
                G += cosine_coefficient(n) * (
                    jump * np.cos(wn * (t - a2))
                    + np.cos(wn * t) * kc + np.sin(wn * t) * ks
                )

        pf[j] = (1.0 - factor) * (1.0 - F) + factor * (g1[j] - G)

    return {'p': pf, 'sigma': sig}


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_settings():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 18,
        'axes.labelsize': 24,
        'legend.fontsize': 20,
        'xtick.labelsize': 19,
        'ytick.labelsize': 19,
    })
    return plt


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper-figure', type=int, choices=(1, 2, 3), default=1)
    p.add_argument('--time', type=float, default=None)
    p.add_argument('--pressure', type=float, default=1000.0)
    p.add_argument('--depth-plotted', type=float, default=100.0)
    p.add_argument('--analytical-modes', type=int, default=64)
    p.add_argument('--out', default='LO_2010_10Hz_ONLY')
    args = p.parse_args()
    if args.time is None:
        args.time = {1: 60.0, 2: 600.0, 3: 3600.0}[args.paper_figure]
    args.freqs = (10.0,)  # Only the 10 Hz response is calculated and plotted.
    if args.time <= 0 or args.depth_plotted <= 0 or args.analytical_modes < 1:
        p.error('Invalid settings.')
    return args


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def plot_results(results, out, t, figure, compared):
    """Original plot layout, using FEM curves only when available."""
    plt = plot_settings()
    target = out / 'PAPER_FIGURES_ONLY'
    target.mkdir(parents=True, exist_ok=True)
    fields = [('p', 'p_Lo_analytical', 'p_numerical', 'Normalized pore pressure, $p/P_0$', figure),
              ('sigma', 'sigma_Lo_analytical', 'sigma_numerical',
               'Normalized total vertical stress, $\\sigma_{zz}/P_0$', figure + 3)]
    for field, ref_key, num_key, xlabel, num in fields:
        fig, axes = plt.subplots(len(results), 1, figsize=(8.9, 4.6*len(results)), squeeze=False)
        for j, f in enumerate(sorted(results)):
            ax = axes[j, 0]
            rows = results[f]
            z = np.array([r['depth_m'] for r in rows])
            ana = np.array([r[ref_key] for r in rows])
            if compared[f]:
                fem = np.array([r[num_key] for r in rows])
                ax.plot(fem, z, '-', lw=2.2, label='Numerical (FEniCSx UPQ)')
            ax.plot(ana, z, ':', lw=2.6, label='Lo (2010), causal dynamic analytical')
            ax.set_ylabel('Depth (m)')
            ax.set_xlabel(xlabel)
            ax.set_title(f'Lo Fig. {num}:  $f={f:g}$ Hz,  $t={t:g}$ s', fontsize=17)
            ax.set_ylim(float(np.max(z)), 0.0)
            if field == 'p':
                ax.set_xlim(-0.25, 1.10)
            else:
                ax.set_xlim(-1.1, 0.25)
            ax.minorticks_on()
            ax.grid(True, ls=':', alpha=0.45, which='major')
            ax.grid(True, ls=':', alpha=0.18, which='minor')
            ax.legend(frameon=False, fontsize=13, loc='best')
        fig.tight_layout()
        suffix = 'FEM_vs_analytical' if all(compared.values()) else 'analytical_only'
        stem = target / f'Lo_Fig{num}_{field}_{suffix}'
        fig.savefig(str(stem) + '.png', dpi=300, bbox_inches='tight')
        fig.savefig(str(stem) + '.pdf', bbox_inches='tight')
        plt.close(fig)
    print('Analytical plots saved.', flush=True)


def main():
    args = arguments()
    out = Path(args.out) / f'Fig{args.paper_figure}_t{args.time:g}s'
    out.mkdir(parents=True, exist_ok=True)
    results, compared = {}, {}
    for f in args.freqs:
        folder = out / f'f_{f:g}Hz'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'Depth_profiles.csv'
        fem_rows = read_csv(path) if path.exists() else []
        if fem_rows:
            z = np.array([float(row['depth_m']) for row in fem_rows])
        else:
            z = np.unique(np.r_[0.0, np.linspace(0.002, 2.0, 65),
                                  np.linspace(2.1, args.depth_plotted, 285)])
        print(f'Analytical solution: {f:g} Hz.', flush=True)
        ref = lo_solution(z, args.time, f, args.pressure, args.analytical_modes)
        profiles = [{'frequency_Hz': f, 'time_s': args.time, 'depth_m': float(depth),
                     'p_Lo_analytical': float(p), 'sigma_Lo_analytical': float(s)}
                    for depth, p, s in zip(z, ref['p'], ref['sigma'])]
        write_csv(folder / 'Analytical_profiles.csv', profiles)
        if fem_rows:
            for row, num in zip(profiles, fem_rows):
                row['p_numerical'] = float(num['p_numerical'])
                row['sigma_numerical'] = float(num['sigma_numerical'])
            write_csv(folder / 'Depth_profiles_comparison.csv', profiles)
            errors = []
            for field in ('p', 'sigma'):
                numerical = np.array([r[f'{field}_numerical'] for r in profiles])
                analytical = np.array([r[f'{field}_Lo_analytical'] for r in profiles])
                rel = np.sqrt(np.trapezoid((numerical-analytical)**2, z)
                              / max(np.trapezoid(analytical**2, z), 1e-14))
                errors.append({'frequency_Hz': f, 'field': field, 'rel_L2_percent': float(100*rel),
                               'max_abs_error_normalized': float(np.max(abs(numerical-analytical)))})
            write_csv(folder / 'Error_metrics.csv', errors)
        results[f] = profiles
        compared[f] = bool(fem_rows)
    plot_results(results, out, args.time, args.paper_figure, compared)
    print(f'Analytical finished. Results: {out}', flush=True)


if __name__ == '__main__':
    main()

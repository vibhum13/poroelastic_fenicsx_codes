# Fully dynamic poroelastic benchmark: 10 Hz harmonic loading

This directory contains **two separate Python scripts** for the saturated poroelastic half-space under a **10 Hz harmonic surface load**, based on the benchmark of Lo (2010):

1. **`dynamic.py`** - the fully dynamic FEniCSx finite element implementation with solid displacement (`u`), pore pressure (`p`), and relative fluid flux (`q`) as the unknowns.
2. **`analytical_lo.py`** - a separate analytical-reference script used to generate depth profiles and compare them with FEM results.

The loading in these scripts is a *rectified cosine* of the form

\[
\sigma_{zz}(0,t) = -P_0 |\cos(2\pi f t)|, \qquad f=10\,\mathrm{Hz}.
\]

The top boundary is drained. In the supplied scripts the initial pore pressure and vertical total stress are set to their stated reference values. The FEM uses a narrow two-dimensional computational strip and a truncation boundary for the half-space. These details, rather than an exact infinite half-space simulation, should be considered when comparing results.

## Files

| File | Role |
| --- | --- |
| `dynamic.py` | Fully dynamic FEniCSx `u-p-q` finite element simulation at 10 Hz; writes numerical pressure/stress depth profiles, an Excel workbook, and visualizations |
| `analytical_lo.py` | Separate Lo (2010) analytical pressure and stress solution; writes analytical profiles and, if the FEM CSV is found, comparison profiles, plots, and error metrics |

## Software

- The dynamic FEM code is written for **FEniCSx / DOLFINx 0.9**, MPI, NumPy, PETSc, UFL, Basix and Matplotlib.
- The analytical script uses **NumPy, SciPy, and Matplotlib**; it does not require the FEniCSx solver to generate analytical-only results.
- The FEM script uses a standard-library Excel writer and does not need `openpyxl` for its `.xlsx` output.

## Step 1 - Run the FEM script

Open a terminal **inside this `dynamic/` directory** and run:

```bash
mpirun -n 4 python3 dynamic.py
```

For a short diagnostic only (not a full validation):

```bash
mpirun -n 4 python3 dynamic.py --quick
```

The default FEM settings include `f = 10 Hz` and `t = 60 s`. The default output location is:

```text
LO_2010_REPRODUCTION_CAUSAL_UPQ_NEW/
└── Fig1_t60s/
    ├── f_10Hz/
    │   ├── Depth_profiles.csv
    │   ├── Time_histories.csv
    │   ├── Parameters.csv
    │   ├── Lo_2010_mixed_UPQ_validation.xlsx
    │   └── VTK/                    # FEM field visualizations
    └── PAPER_FIGURES_ONLY/       # FEM-only pressure and stress plots
```

## Step 2 - Run the separate analytical script and compare

**Important:** The two source files have different default output-directory names. Pass `--out` as below so the analytical script finds `Depth_profiles.csv` from the FEM run.

```bash
python3 analytical_lo.py --out LO_2010_REPRODUCTION_CAUSAL_UPQ_NEW
```

The analytical script then uses **exactly the same depth coordinates** as the FEM file and generates these outputs:

```text
LO_2010_REPRODUCTION_CAUSAL_UPQ_NEW/
└── Fig1_t60s/
    ├── f_10Hz/
    │   ├── Analytical_profiles.csv
    │   ├── Depth_profiles_comparison.csv
    │   └── Error_metrics.csv
    └── PAPER_FIGURES_ONLY/
        ├── Lo_Fig1_p_FEM_vs_analytical.png
        ├── Lo_Fig1_p_FEM_vs_analytical.pdf
        ├── Lo_Fig4_sigma_FEM_vs_analytical.png
        └── Lo_Fig4_sigma_FEM_vs_analytical.pdf
```

These two comparisons display **normalized pore pressure** `p/P0` and **normalized total vertical stress** `sigma_zz/P0` against depth, respectively.

For analytical profiles **without** FEM data, run:

```bash
python3 analytical_lo.py
```

This standalone command uses its own default output directory, `LO_2010_10Hz_ONLY/`, and creates analytical-only plots. It does not generate a FEM comparison unless it finds `Depth_profiles.csv` in that directory.

### Observation time and options

Both default to `60 s`. If you run the FEM using `--time 10`, also invoke the analytical code using `--time 10 --out LO_2010_REPRODUCTION_CAUSAL_UPQ_NEW`. The analytical code must use the same forcing amplitude and observed depth as the FEM run for consistent comparisons.

## Notes on interpretation

- `dynamic.py` uses CG1-DG0-BDM1 finite element fields. The analytical comparison uses the computed FEM pressure and stress profiles, rather than editing the numerical solution.
- The reference script computes the 10 Hz case **only**. No 1 Hz or 100 Hz profile is produced by this version.
- The numerical and analytical profiles are related validation outputs but come from **different scripts**; do not interpret the analytical-only plot as an FEM validation figure.
- The repository contains reproducible code and necessary inputs, not precomputed or experimentally validated data. Ensure the comparisons are assessed before using them in a publication.

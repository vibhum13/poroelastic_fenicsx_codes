# FEniCSx codes for poroelastic benchmark problems

Open-source finite element implementations of **dynamic, quasi-static, and steady-state poroelastic benchmark problems**, with analytical or semi-analytical comparisons. The dynamic example uses **10 Hz harmonic surface loading** and reports **pore-pressure and total vertical-stress profiles** for comparison with the analytical solution of Lo (2010).

## Contents

```text
fenicsx-poroelastic-codes/
├── README.md
├── CITATION.cff
├── LICENSE
├── .gitignore
├── dynamic/
│   ├── dynamic.py             # Fully dynamic 10 Hz FEM, u-p-q
│   ├── analytical_lo.py       # Separate Lo (2010) analytical reference
│   └── README.md
├── quasistatic/
│   ├── mandel.py              # Mandel consolidation
│   └── README.md
└── steady_state/
    ├── steady.py              # Steady poroelastic annulus
    ├── tri.geo                # Gmsh geometry
    ├── mesh.msh               # Mesh source
    ├── mesh.xdmf              # Cell mesh / tags
    ├── mesh.h5                # Required XDMF data
    ├── mt.xdmf                # Facet tags
    ├── mt.h5                  # Required facet data
    └── README.md
```

## Models

| Directory | Benchmark | Unknown fields | Main comparisons |
| --- | --- | --- | --- |
| [`dynamic/`](dynamic/) | Lo (2010) saturated poroelastic half-space, 10 Hz rectified harmonic load | Solid displacement `u`, pore pressure `p`, relative fluid flux `q` | Numerical and analytical profiles of **pore pressure** and **total vertical stress** versus depth at 60 s |
| [`quasistatic/`](quasistatic/) | Mandel consolidation | Solid displacement and pore pressure | Numerical vs analytical transient pressure |
| [`steady_state/`](steady_state/) | Steady coupled poroelastic annulus | Solid displacement and pore pressure | Analytical vs numerical radial displacement; computed pressure and flow |

## Requirements

- Python 3, NumPy, Matplotlib, MPI (`mpi4py`), PETSc (`petsc4py`), UFL, Basix, and DOLFINx / FEniCSx.
- The **dynamic FEM script** targets **FEniCSx 0.9**; the quasi-static and steady-state scripts were originally documented for **FEniCSx 0.8**. Other versions may require API changes.
- SciPy is additionally required for the separate dynamic analytical script (`analytical_lo.py`).
- The Mandel script uses Pandas; see the corresponding script and folder README for details.
- For the steady-state example, keep `.xdmf` and associated `.h5` mesh files together.

These files are provided as individual research examples, not as a single installed Python package.

## Running the dynamic 10 Hz benchmark

From the repository root:

```bash
cd dynamic
mpirun -n 4 python3 dynamic.py
# When the FEM run is complete, evaluate the analytical reference
# in the same output folder to obtain overlaid pressure/stress plots:
python3 analytical_lo.py --out LO_2010_REPRODUCTION_CAUSAL_UPQ_NEW
```

The first script writes `Depth_profiles.csv`, an Excel workbook, and FEM figures. The second script reads that same FEM profile (if it exists) and adds `Analytical_profiles.csv`, `Depth_profiles_comparison.csv`, error metrics, and **numerical-versus-analytical pore-pressure and vertical-stress plots**.

Both scripts use the 10 Hz case by default; the FEM default is `t = 60 s`. For diagnostic short runs, see the detailed [dynamic README](dynamic/README.md). **Run the analytical script with the same observation time as the FEM results** if you override `--time`.

## Running the other benchmarks

```bash
cd quasistatic
python3 mandel.py
```

```bash
cd steady_state
python3 steady.py
```

Run each command from its own folder so relative paths resolve correctly. Detailed instructions and output descriptions are in each folder's `README.md`.

## Reproducibility and outputs

Default simulation outputs are generated when scripts run and are omitted from Git using [`.gitignore`](.gitignore). Required input meshes under `steady_state/` **are intentionally retained**, including `.h5` files. No numerical result is claimed here unless the corresponding script has been run and checked in a compatible environment.

## Citation

To cite this software, use GitHub's **Cite this repository** menu, which reads [`CITATION.cff`](CITATION.cff). For an archived, version-specific citation, cite the DOI assigned to the corresponding GitHub release by Zenodo. A DOI can be added to the README after Zenodo creates it. Please also cite the original analytical benchmark(s) when using them.

## License

Distributed under the [MIT License](LICENSE).

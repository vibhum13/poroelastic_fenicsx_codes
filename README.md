# FEniCSx implementations of poroelastic benchmark problems

Open-source research code for **dynamic**, **quasi-static**, and **steady-state** poroelastic benchmark problems implemented with FEniCSx. The repository contains numerical models and, where implemented, comparisons with analytical reference solutions.

## Repository contents

| Folder | Main script | Problem and numerical formulation |
| --- | --- | --- |
| [`dynamic/`](dynamic/) | [`dynamic.py`](dynamic/dynamic.py) |Transient dynamic response of saturated poroelastic media under harmonic surface loading, formulated in terms of solid displacement, pore pressure, and relative fluid flux (`u-p-q`). The implementation includes numerical verification against the analytical solution of Lo (2010), with comparisons of pore-pressure and vertical-stress profiles.
| [`quasistatic/`](quasistatic/) | [`mandel.py`](quasistatic/mandel.py) | Mandel consolidation benchmark, solved using a coupled displacement-pressure formulation. |
| [`steady state/`](steady%20state/) | [`steady.py`](steady%20state/steady.py) | Steady-state annular poroelastic problem with prescribed fluid flux at the inner boundary; numerical radial deformation is compared with an analytical solution. |

The three folders have additional README files describing the respective benchmark setup. The programs are **research examples**, not a general-purpose installation package.

## Requirements

- Python with a compatible FEniCSx installation (`dolfinx`, `basix`, `ufl`)
- `mpi4py`, `petsc4py` and PETSc (with an available linear solver compatible with the scripts)
- `numpy` and `matplotlib`
- `pandas` for the Mandel benchmark
- Gmsh-format/XDMF mesh files in the `steady state/` folder (already included in this repository)

**Version compatibility:** The existing benchmark README files identify FEniCSx **0.8** as their target. FEniCSx APIs can change between releases; do not assume that these scripts run unchanged with later versions. Check your own environment before interpreting computed results.

## Running the benchmarks

Run each script **from its own directory** so relative input/output paths work as intended.

**Dynamic benchmark**

```bash
cd dynamic
python3 dynamic.py
```

**Quasi-static Mandel benchmark**

```bash
cd quasistatic
python3 mandel.py
```

**Steady-state annular benchmark**

```bash
cd 'steady state'
python3 steady.py
```

For parallel execution, scripts that support MPI may be run using a suitable `mpirun -n N python3 script.py` command. Verify MPI compatibility and available memory in your environment before increasing the number of processes.

The steady-state mesh uses `mesh.xdmf` and `mt.xdmf`, which refer to the accompanying HDF5 (`.h5`) datasets. **Keep these mesh files together**; the `.h5` files in that folder are required inputs, not disposable output.

## Outputs and reproducibility

The scripts write numerical solution fields and/or validation plots to their configured output directories. Generated figures, VTK series and simulation results are intentionally excluded from source-control by `.gitignore` where appropriate.

For reproducible use, report the specific GitHub release/Zenodo DOI, the FEniCSx and PETSc versions, material/mesh parameters, and any modifications made to the scripts. See the per-benchmark README files and script headers for model details.

## Citation

Use GitHub's **Cite this repository** option (generated from [`CITATION.cff`](CITATION.cff)). After a GitHub release has been archived through Zenodo, cite the DOI associated with the **specific software version used**. The DOI can be added to this section after the first Zenodo release.

If a corresponding journal article is published, its citation can be added here separately; the code citation and article citation are not automatically the same.

## License

Distributed under the [MIT License](LICENSE). This covers the repository's original code and documentation; third-party components retain their own licenses.

## Author

Vibhum Mahajan — Indian Institute of Technology Roorkee.

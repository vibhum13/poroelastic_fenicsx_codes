# Quasi-static Mandel poroelastic analysis

This folder contains the FEniCSx code for the quasi-static Mandel poroelastic problem.

The code solves the coupled displacement-pressure response of a poroelastic domain under quasi-static loading. The formulation neglects inertial effects and evaluates the time-dependent consolidation response of the porous medium.

## File

- `mandel.py`: Quasi-static Mandel poroelastic FEniCSx code.

## How the code works

The code defines the mesh, material properties, mixed finite element space, boundary conditions, quasi-static weak form, and time-stepping loop. At each time step, the coupled displacement-pressure system is solved, the solution variables are updated, and the results are saved for post-processing and validation.

## Additional Python dependencies

The script imports NumPy, Pandas, and Matplotlib along with FEniCSx and its MPI/PETSc dependencies.

## How to run

This script was originally documented for FEniCSx version 0.8. It may require minor changes if a different FEniCSx version is used.

Open the terminal in the same folder as `mandel.py` and run the code using either of the following commands:

```bash
python mandel.py
```

or:

```bash
python3 mandel.py
```
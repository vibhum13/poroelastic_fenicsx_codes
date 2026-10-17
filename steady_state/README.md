# Steady-state poroelastic annulus analysis

This folder contains the FEniCSx code for the steady-state poroelastic annulus problem.

The code solves a coupled displacement-pressure problem in an annular domain with an inner circular cavity and an outer boundary. A fluid flux is applied at the inner boundary, while zero displacement and zero pore pressure are imposed at the outer boundary. The numerical radial deformation is compared with the analytical solution.

## Files

- `steady.py`: Steady-state poroelastic FEniCSx code.
- `mesh.xdmf`: Mesh file.
- `mt.xdmf`: Boundary marker file.
- `mesh.h5` and `mt.h5`: Required mesh/marker data files linked with the XDMF files.

## How the code works

The code reads the mesh and boundary markers, defines the mixed displacement-pressure finite element space, applies the boundary conditions, formulates the coupled steady-state weak form, and solves the monolithic linear system. The displacement, pressure, Darcy velocity, analytical comparison data, and validation plot are saved in the `results` folder.

## How to run

This script was originally documented for FEniCSx version 0.8. It may require minor changes if a different FEniCSx version is used.

Keep `steady.py`, `mesh.xdmf`, `mt.xdmf`, `mesh.h5`, and `mt.h5` in the same folder.

Open the terminal in the same folder as `steady.py` and run the code using either of the following commands:

```bash
python steady.py
```

or:

```bash
python3 steady.py
```
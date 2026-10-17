# =============================================================================
# Steady-State Poroelastic Annulus Problem
# =============================================================================
# This code solves a coupled steady-state poroelastic problem in an annular domain
# with an inner circular cavity and an outer circular boundary.
#
# The model uses a mixed finite element formulation with displacement and pore
# pressure as primary unknowns. A fluid flux is applied at the inner boundary,
# while zero displacement and zero pore pressure are imposed at the outer boundary.
#
# The coupled mechanical equilibrium and steady Darcy pressure equation are solved
# monolithically in FEniCSx. The numerical radial deformation is then compared
# with the corresponding analytical solution.
# =============================================================================

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
import matplotlib.pyplot as plt
import os
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem, io, geometry
from dolfinx.fem import Function, Constant
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile


# =============================================================================
# MPI communicator and output folder for all result files
# =============================================================================
comm = MPI.COMM_WORLD
results_dir = "results"

if comm.rank == 0:
    os.makedirs(results_dir, exist_ok=True)

comm.barrier()


# =============================================================================
# Read the domain mesh and cell markers converted from Gmsh
# =============================================================================
with io.XDMFFile(comm, "mesh.xdmf", "r") as xdmf:
    msh = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(msh, name="Grid")


# =============================================================================
# Create mesh topology connectivity required for reading and using facet tags
# =============================================================================
gdim = msh.topology.dim
fdim = gdim - 1

msh.topology.create_entities(fdim)
msh.topology.create_connectivity(fdim, gdim)
msh.topology.create_connectivity(gdim, fdim)


# =============================================================================
# Read boundary facet markers from mt.xdmf
# =============================================================================
with io.XDMFFile(comm, "mt.xdmf", "r") as xdmf:
    ft = xdmf.read_meshtags(msh, name="Grid")


# =============================================================================
# Define integration measures using cell and facet markers
# =============================================================================
dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

# Boundary markers defined in the Gmsh physical curves
inner_marker = 1
outer_marker = 2

inner_facets = ft.find(inner_marker)
outer_facets = ft.find(outer_marker)


# =============================================================================
# Define mixed finite element space for displacement and pressure
# =============================================================================
p_u = element("Lagrange", msh.topology.cell_name(), 2, shape=(msh.topology.dim,), dtype=default_real_type)
p_p = element("Lagrange", msh.topology.cell_name(), 1, dtype=default_real_type)

W_elem = mixed_element([p_u, p_p])
W = fem.functionspace(msh, W_elem)


# =============================================================================
# Define trial and test functions for the coupled problem
# =============================================================================
(u, p) = ufl.TrialFunctions(W)
(q, v) = ufl.TestFunctions(W)


# =============================================================================
# Create zero displacement and zero pressure functions for outer boundary BCs
# =============================================================================
W0 = W.sub(0)
Q0, _ = W0.collapse()

u_bc = Function(Q0)
u_bc.x.array[:] = 0.0
u_bc.x.scatter_forward()

W1 = W.sub(1)
Q1, _ = W1.collapse()

p_bc = Function(Q1)
p_bc.x.array[:] = 0.0
p_bc.x.scatter_forward()


# =============================================================================
# Locate mixed-space DOFs on the outer boundary using XDMF facet markers
# =============================================================================
dofs_u = fem.locate_dofs_topological((W0, Q0), fdim, outer_facets)
dofs_p = fem.locate_dofs_topological((W1, Q1), fdim, outer_facets)

outer_boundary_displacement = fem.dirichletbc(u_bc, dofs_u, W0)
outer_boundary_pressure = fem.dirichletbc(p_bc, dofs_p, W1)

bcs = [outer_boundary_displacement, outer_boundary_pressure]


# =============================================================================
# Define elastic material parameters
# =============================================================================
E = 1.0
nu = 0.4 / 1.4
G = E / (2.0 * (1.0 + nu))


# =============================================================================
# Small-strain tensor
# =============================================================================
def epsilon(u):
    return ufl.sym(ufl.grad(u))


# =============================================================================
# Poroelastic stress tensor with pore-pressure contribution
# =============================================================================
def sigma(u, p):
    lam = (2.0 * G * nu) / (1.0 - 2.0 * nu)
    return 2.0 * G * ufl.sym(ufl.grad(u)) + lam * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(len(u)) - p * ufl.Identity(len(u))


Stress = sigma(u, p)


# =============================================================================
# Mechanical equilibrium weak form
# =============================================================================
B = Constant(msh, np.array((0.0, 0.0), dtype=PETSc.ScalarType))

a1 = ufl.inner(Stress, ufl.grad(q)) * dx
L1 = ufl.inner(B, q) * dx


# =============================================================================
# Steady-state pressure equation with inner-boundary flux
# =============================================================================
f = Constant(msh, PETSc.ScalarType(0.0))
k = Constant(msh, PETSc.ScalarType(1.0))

pressure_drop = 0.33
r1 = 0.001
r2 = 1.0

g1 = Constant(msh, PETSc.ScalarType(-pressure_drop / (r1 * (np.log(r1) - np.log(r2)))))

A1 = 2.0 * np.pi * r1
A2 = 2.0 * np.pi * r2
g2 = -g1.value * (A1 / A2)

a2 = k * ufl.inner(ufl.grad(p), ufl.grad(v)) * dx
L2 = f * v * dx + g1 * v * ds(inner_marker)


# =============================================================================
# Assemble the coupled weak form
# =============================================================================
a = a1 + a2
L = L1 + L2


# =============================================================================
# Solve the coupled linear system
# =============================================================================
w_coupled = fem.Function(W)

problem = LinearProblem(a, L, bcs=bcs, u=w_coupled, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
w_coupled = problem.solve()


# =============================================================================
# Split the mixed solution into displacement and pressure fields
# =============================================================================
u_sol = w_coupled.sub(0).collapse()
p_sol = w_coupled.sub(1).collapse()

u_sol.name = "displacement"
p_sol.name = "pressure"


# =============================================================================
# Save displacement and pressure fields
# =============================================================================
# Important:
# u_sol is P2, while mesh geometry is usually P1.
# XDMF requires the output function degree to match the mesh degree.
# Therefore, displacement is interpolated to P1 only for writing.
# The original P2 u_sol is still used for numerical evaluation.

u_out_element = element("Lagrange", msh.topology.cell_name(), 1, shape=(msh.topology.dim,), dtype=default_real_type)
V_u_out = fem.functionspace(msh, u_out_element)

u_out = fem.Function(V_u_out)
u_out.name = "displacement"

u_expr = fem.Expression(u_sol, V_u_out.element.interpolation_points())
u_out.interpolate(u_expr)
u_out.x.scatter_forward()

with XDMFFile(comm, os.path.join(results_dir, "displacement.xdmf"), "w") as file:
    file.write_mesh(msh)
    file.write_function(u_out)

with XDMFFile(comm, os.path.join(results_dir, "pressure.xdmf"), "w") as file:
    file.write_mesh(msh)
    file.write_function(p_sol)


# =============================================================================
# Project Darcy velocity field from the pressure gradient
# =============================================================================
velocity_element = element("Lagrange", msh.topology.cell_name(), 1, shape=(msh.topology.dim,), dtype=default_real_type)
V = fem.functionspace(msh, velocity_element)

w_vel = Function(V)
w_expr = fem.Expression(-k * ufl.grad(p_sol), V.element.interpolation_points())

w_vel.interpolate(w_expr)
w_vel.x.scatter_forward()
w_vel.name = "velocity"

with XDMFFile(comm, os.path.join(results_dir, "velocity.xdmf"), "w") as file:
    file.write_mesh(msh)
    file.write_function(w_vel)


# =============================================================================
# Analytical radial displacement solution
# =============================================================================
r_0 = 0.001
r_max = 1.0

r = np.arange(r_0, r_max, 0.0001)

Gamma = 0.4
q_flux = 0.036806

B1 = -(1.0 - Gamma) * (q_flux * (1.0 + (1.0 + Gamma) * np.log(r_0)) / (2.0 * ((r_0**2) * (1.0 + Gamma) + (1.0 - Gamma)))) + ((1.0 + Gamma) / 2.0) * q_flux * np.log(r_0)
B2 = -((1.0 - Gamma) * (q_flux * r_0**2 * (1.0 + (1.0 + Gamma) * np.log(r_0)) / (2.0 * ((r_0**2) * (1.0 + Gamma) + (1.0 - Gamma)))))

u_analytic = (-q_flux * r * np.log(r)) / 2.0 + ((2.0 * B1 + q_flux) * r) / (2.0 * (1.0 + Gamma)) + B2 / ((1.0 - Gamma) * r)


# =============================================================================
# Evaluate a finite element function at selected physical points
# =============================================================================
def evaluate_function_on_points(function, points):
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    valid_points = []
    values = []

    for i, point in enumerate(points):
        cells = colliding_cells.links(i)

        if len(cells) > 0:
            value = function.eval(point, cells[0])
            valid_points.append(point)
            values.append(value)

    if len(valid_points) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, msh.topology.dim), dtype=np.float64)

    return np.array(valid_points, dtype=np.float64), np.array(values, dtype=np.float64)


# =============================================================================
# Sample numerical displacement along the positive x-axis
# =============================================================================
r_num = np.linspace(r_0 + 1e-6, r_max - 1e-6, 1000)
y_sample = 1e-10

points = np.zeros((len(r_num), 3), dtype=np.float64)
points[:, 0] = r_num
points[:, 1] = y_sample
points[:, 2] = 0.0

valid_points, u_values = evaluate_function_on_points(u_sol, points)


# =============================================================================
# Convert numerical displacement vector into radial displacement
# =============================================================================
if len(valid_points) > 0:
    r_valid = np.sqrt(valid_points[:, 0] ** 2 + valid_points[:, 1] ** 2)
    u_radial_num = u_values[:, 0] * valid_points[:, 0] / r_valid + u_values[:, 1] * valid_points[:, 1] / r_valid
    local_compare = np.column_stack([r_valid, u_radial_num])
else:
    local_compare = np.empty((0, 2), dtype=np.float64)


# =============================================================================
# Gather numerical line data from all MPI ranks
# =============================================================================
all_compare = comm.gather(local_compare, root=0)


# =============================================================================
# Plot and save analytical-numerical comparison
# =============================================================================
if comm.rank == 0:
    numerical_data = np.vstack([arr for arr in all_compare if arr.size > 0])

    sort_id = np.argsort(numerical_data[:, 0])
    numerical_data = numerical_data[sort_id]

    r_valid = numerical_data[:, 0]
    u_radial_num = numerical_data[:, 1]

    _, unique_id = np.unique(np.round(r_valid, 12), return_index=True)
    r_valid = r_valid[unique_id]
    u_radial_num = u_radial_num[unique_id]

    plt.figure()
    plt.plot(r_valid, u_radial_num, color="lightskyblue", linewidth=3.0, alpha=0.75)
    plt.plot(r, u_analytic, "r--", linewidth=2.0)
    plt.xlabel("Radius")
    plt.ylabel("Radial deformation")
    plt.title("Analytical vs numerical radial deformation")
    plt.grid(True)
    plt.legend(["Numerical", "Analytical"], loc="best")
    plt.savefig(os.path.join(results_dir, "analytical_vs_numerical_deformation.png"), dpi=600, bbox_inches="tight")
    plt.show()

    out_analytic = np.column_stack([r, u_analytic])
    np.savetxt(os.path.join(results_dir, "u_analytic_vs_r.csv"), out_analytic, delimiter=",", header="r,u_analytic", comments="")

    out_numerical = np.column_stack([r_valid, u_radial_num])
    np.savetxt(os.path.join(results_dir, "u_numerical_vs_r.csv"), out_numerical, delimiter=",", header="r,u_radial_numerical", comments="")

    u_num_interp = np.interp(r, r_valid, u_radial_num)
    out_compare = np.column_stack([r, u_analytic, u_num_interp])
    np.savetxt(os.path.join(results_dir, "u_analytical_numerical_comparison.csv"), out_compare, delimiter=",", header="r,u_analytic,u_numerical_interpolated", comments="")

    print("Wrote results/displacement.xdmf")
    print("Wrote results/pressure.xdmf")
    print("Wrote results/velocity.xdmf")
    print("Wrote results/u_analytic_vs_r.csv")
    print("Wrote results/u_numerical_vs_r.csv")
    print("Wrote results/u_analytical_numerical_comparison.csv")
    print("Wrote results/analytical_vs_numerical_deformation.png")

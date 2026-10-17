# ==================================================================================================
# Quasi-static poroelastic Mandel problem validation in FEniCSx.
# This code solves the coupled displacement-pore-pressure formulation for the Mandel consolidation
# benchmark under quasi-static mechanical equilibrium and transient pore-pressure dissipation.
# The numerical pore pressure is extracted at the point (x, y) = (12.5, 0).
# The corresponding analytical Mandel pressure is evaluated at x' = |12.5 - width/2|.
# Numerical and analytical pore-pressure histories are compared in one validation plot.
# ==================================================================================================

from mpi4py import MPI
import numpy as np, pandas as pd, os, matplotlib.pyplot as plt
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from basix.ufl import element, mixed_element
from dolfinx import fem, mesh, geometry
from dolfinx.mesh import CellType
from dolfinx.fem import Function, Constant
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import VTKFile
import ufl
from ufl import Measure, FacetNormal, nabla_div, dot, inner, grad, derivative

results_dir = "mandel_numerical_analytical_comparison"
os.makedirs(results_dir, exist_ok=True)

# ============================================================
# Mesh
# Define a 2D rectangular Mandel domain using quadrilateral elements.
# ============================================================
nx, ny = 40, 20
width, Height = 20.0, 10.0
msh = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[0.0, 0.0], [width, Height]]), [nx, ny], cell_type=CellType.quadrilateral)
comm, rank = msh.comm, msh.comm.rank


# Boundary indicator functions for locating external facets.
def top(x): return np.isclose(x[1], Height)
def bottom(x): return np.isclose(x[1], 0.0)
def left(x): return np.isclose(x[0], 0.0)
def right(x): return np.isclose(x[0], width)

# Locate boundary facets.
fdim = msh.topology.dim - 1
left_facets = mesh.locate_entities_boundary(msh, fdim, left)
right_facets = mesh.locate_entities_boundary(msh, fdim, right)
top_facets = mesh.locate_entities_boundary(msh, fdim, top)
bottom_facets = mesh.locate_entities_boundary(msh, fdim, bottom)

# Assign physical tags to boundaries: bottom = 1, right = 2, top = 3, left = 4.
marked_facets = np.hstack([top_facets, bottom_facets, left_facets, right_facets])
marked_values = np.hstack([np.full_like(top_facets, 3), np.full_like(bottom_facets, 1), np.full_like(left_facets, 4), np.full_like(right_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(msh, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

# ============================================================
# Time and material parameters
# ============================================================
# Define time-step size for transient pressure dissipation.
t, Tf, num_steps = 0.0, 10.0, 500
dt = (Tf - t) / num_steps

# Define material parameters used in the Mandel validation problem.
E_value, nu_value, rho_f_value, rho_s_value = 1.0e9, 0.25, 1000.0, 2100.0
mu_f_value, hydraulic_conductivity_value, alpha_value, phi_value = 1.0e-3, 1.0e-4, 1.0, 0.3
Kf_value, Ks_value, g_value = 2.0e9, 1.0e10, 9.81

# Convert material constants to FEniCSx Constants.
E, nu = Constant(msh, ScalarType(E_value)), Constant(msh, ScalarType(nu_value))
rhol, rhos = Constant(msh, ScalarType(rho_f_value)), Constant(msh, ScalarType(rho_s_value))
mul, conductivity = Constant(msh, ScalarType(mu_f_value)), Constant(msh, ScalarType(hydraulic_conductivity_value))
beta, epsilonl = Constant(msh, ScalarType(alpha_value)), Constant(msh, ScalarType(phi_value))
Kf, Ks = Constant(msh, ScalarType(Kf_value)), Constant(msh, ScalarType(Ks_value))

# Calculate Lame parameters.
lambda_value = E_value * nu_value / ((1.0 + nu_value) * (1.0 - 2.0 * nu_value))
mu_value = E_value / (2.0 * (1.0 + nu_value))
lambda_m, mu = Constant(msh, ScalarType(lambda_value)), Constant(msh, ScalarType(mu_value))

# Convert hydraulic conductivity to intrinsic permeability and define storage coefficient.
kappa_value = hydraulic_conductivity_value * mu_f_value / (rho_f_value * g_value)
kepsilon = Constant(msh, ScalarType(kappa_value))
S_value = phi_value / Kf_value + (1.0 - phi_value) / Ks_value
S = Constant(msh, ScalarType(S_value))

# ============================================================
# Mandel parameters
# ============================================================
# Calculate analytical Mandel parameters used for validation and pressure scaling.
F_total, a_mandel = 100000.0, width / 2.0
M_value = 1.0 / S_value
Ku_value = lambda_value + 2.0 * mu_value / 3.0 + alpha_value**2 * M_value
B_value = alpha_value * M_value / Ku_value
nu_u_value = (3.0 * nu_value + alpha_value * B_value * (1.0 - 2.0 * nu_value)) / (3.0 - alpha_value * B_value * (1.0 - 2.0 * nu_value))
nu_u = Constant(msh, ScalarType(nu_u_value))
pinit = B_value * (1.0 + nu_u_value) * F_total / (3.0 * a_mandel)
c_value = (kappa_value / mu_f_value) / (S_value + alpha_value**2 / (lambda_value + 2.0 * mu_value))

# Print calculated parameters on rank 0 only.
if rank == 0:
    print(f"lambda = {lambda_value:.6e} Pa")
    print(f"mu     = {mu_value:.6e} Pa")
    print(f"S      = {S_value:.6e} 1/Pa")
    print(f"M      = {M_value:.6e} Pa")
    print(f"Ku     = {Ku_value:.6e} Pa")
    print(f"B      = {B_value:.6f}")
    print(f"nu_u   = {nu_u_value:.6f}")
    print(f"pinit  = {pinit:.6f} Pa")
    print(f"kappa  = {kappa_value:.6e} m^2")
    print(f"c      = {c_value:.9f} m^2/s")

# ============================================================
# Loading, measures, and function space
# ============================================================
# Define the compressive traction and integration measures.
T_value = -10000.0
T = Constant(msh, ScalarType(T_value))
ddx = ufl.Measure("dx", domain=msh)
ds = Measure("ds", domain=msh, subdomain_data=facet_tag)
normal = FacetNormal(msh)

# Define mixed finite element space: displacement uses quadratic Lagrange and pressure uses linear Lagrange.
displacement_element = element("Lagrange", msh.topology.cell_name(), 2, shape=(msh.topology.dim,))
pressure_element = element("Lagrange", msh.topology.cell_name(), 1)
MS = fem.functionspace(msh, mixed_element([displacement_element, pressure_element]))

# ============================================================
# Boundary conditions
# ============================================================
# Apply vertical displacement constraint at the bottom and drained pressure conditions at the left/right boundaries.
bottom_dofs = fem.locate_dofs_topological(MS.sub(0).sub(1), fdim, bottom_facets)
right_dofs = fem.locate_dofs_topological(MS.sub(1), fdim, right_facets)
left_dofs = fem.locate_dofs_topological(MS.sub(1), fdim, left_facets)

bottom_boundary_displacement = fem.dirichletbc(ScalarType(0.0), bottom_dofs, MS.sub(0).sub(1))
right_boundary_pressure = fem.dirichletbc(ScalarType(0.0), right_dofs, MS.sub(1))
left_boundary_pressure = fem.dirichletbc(ScalarType(0.0), left_dofs, MS.sub(1))
bcs = [left_boundary_pressure, right_boundary_pressure, bottom_boundary_displacement]

# ============================================================
# Initial condition
# ============================================================
# Define solution fields: X0 is the current unknown and Xn stores the previous time-step solution.
X0, Xn = Function(MS), Function(MS)

# Analytical initial displacement field used to initialize the previous displacement state.
def init_displacement(x):
    ux = (T_value * nu_u_value * x[0]) / (2.0 * mu_value * width)
    uy = -(T_value * (1.0 - nu_u_value) * x[1]) / (2.0 * mu_value * width)
    return np.array([ux, uy])

# Interpolate the initial displacement/pressure field and insert it into the part of Xn
Un, Un_to_MS = MS.sub(0).collapse()
FUn_ = Function(Un)
FUn_.interpolate(init_displacement)
Xn.x.array[Un_to_MS] = FUn_.x.array
Xn.x.scatter_forward()

Pn, Pn_to_MS = MS.sub(1).collapse()
FPn_ = Function(Pn)
with FPn_.vector.localForm() as initial_local: initial_local.set(ScalarType(pinit))
Xn.x.array[Un_to_MS] = FUn_.x.array
Xn.x.scatter_forward()


# ============================================================
# Weak form
# ============================================================
# Define small-strain tensor and total stress for the displacement-pressure formulation.
def epsilon(u): return ufl.sym(ufl.grad(u))
def sigma(u, p): return lambda_m * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2.0 * mu * epsilon(u) - p * ufl.Identity(len(u))

# Split mixed unknowns and previous-time fields.
u, p = ufl.split(X0)
u_n, p_n = ufl.split(Xn)
v, q = ufl.TestFunctions(MS)

# Mass balance: volumetric strain rate, Darcy flow term, and pressure storage term.
F = (1.0 / dt) * nabla_div(u - u_n) * q * ddx + (kepsilon / mul) * dot(grad(p), grad(q)) * ddx + (S / dt) * (p - p_n) * q * ddx

# Quasi-static mechanical equilibrium with top traction loading.
F += inner(grad(v), sigma(u, p)) * ddx - T * inner(v, normal) * ds(3)

# Compute Jacobian and define nonlinear problem.
dX0 = ufl.TrialFunction(MS)
J = derivative(F, X0, dX0)
Problem = NonlinearProblem(F, X0, bcs=bcs, J=J)

# Newton solver for the coupled nonlinear residual.
solver = NewtonSolver(MPI.COMM_WORLD, Problem)
solver.atol, solver.rtol, solver.convergence_criterion = 1.0e-6, 1.0e-8, "incremental"

ksp = solver.krylov_solver
opts = PETSc.Options()
prefix = ksp.getOptionsPrefix()
opts[f"{prefix}ksp_type"], opts[f"{prefix}pc_type"] = "preonly", "lu"
ksp.setFromOptions()

# ============================================================
# Analytical Mandel solution
# ============================================================
# Compute positive roots of the Mandel transcendental equation.
def compute_mandel_roots(nu, nu_u, n_roots):
    roots, coeff = [], (1.0 - nu) / (nu_u - nu)
    for n in range(n_roots):
        left, right = n * np.pi + 1.0e-10, n * np.pi + 0.5 * np.pi - 1.0e-10
        for _ in range(100):
            mid = 0.5 * (left + right)
            if (np.tan(left) - coeff * left) * (np.tan(mid) - coeff * mid) <= 0.0: right = mid
            else: left = mid
        roots.append(0.5 * (left + right))
    return np.array(roots)

# Evaluate analytical Mandel pore pressure at a specified position and time array.
def mandel_pressure(times, x_from_center, a, beta_roots, p0, c):
    pressure = np.zeros_like(times, dtype=float)
    for it, time_value in enumerate(times):
        s = 0.0
        for beta_n in beta_roots:
            s += (np.sin(beta_n) / (beta_n - np.sin(beta_n) * np.cos(beta_n))) * (np.cos(beta_n * x_from_center / a) - np.cos(beta_n)) * np.exp(-beta_n**2 * c * time_value / a**2)
        pressure[it] = 2.0 * p0 * s
    return pressure

# Calculate Mandel roots used in the analytical series solution.
beta_ns = compute_mandel_roots(nu_value, nu_u_value, 80)
if rank == 0:
    print("First 12 Mandel roots:")
    print(beta_ns[:12])

# ============================================================
# Point evaluation at numerical point (12.5, 0)
# ============================================================
# Create a bounding-box tree for locating the element containing the evaluation point.
def get_tree(domain):
    try: return geometry.bb_tree(domain, domain.topology.dim)
    except AttributeError: return geometry.BoundingBoxTree(domain, domain.topology.dim)

bb_tree = get_tree(msh)

# Evaluate pressure at a poin
def evaluate_pressure_at_point(p_h, x_eval, y_eval):
    for y_try in [y_eval, y_eval + 1.0e-12, y_eval + 1.0e-10]:
        point = np.array([[x_eval, y_try, 0.0]], dtype=np.float64)
        cell_candidates = geometry.compute_collisions_points(bb_tree, point)
        colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, point)
        cells = colliding_cells.links(0)
        if len(cells) > 0: return float(np.asarray(p_h.eval(point, np.array([cells[0]], dtype=np.int32))).reshape(-1)[0])
    return np.nan

# Gather valid point value from MPI ranks to rank 0.
def gather_value(local_value):
    values = comm.gather(local_value, root=0)
    if rank == 0:
        for value in values:
            if np.isfinite(value): return float(value)
        return np.nan
    return None


# Numerical point and corresponding analytical coordinate measured from the centreline.
x_num, y_num = 12.5, 0.0
x_analytical = abs(x_num - width / 2.0)

if rank == 0:
    print(f"Numerical point       : x = {x_num}, y = {y_num}")
    print(f"Analytical coordinate : x' = {x_analytical}")

# ============================================================
# Output files
# ============================================================
# Create VTK writers for displacement and pressure fields.
file_u = VTKFile(msh.comm, os.path.join(results_dir, "disp.pvd"), "w")
file_p = VTKFile(msh.comm, os.path.join(results_dir, "press.pvd"), "w")
file_u.write_mesh(msh)
file_p.write_mesh(msh)

# Store initial numerical pressure value at the selected point.
time_values, p_num_values = [], []
p_h = X0.split()[1].collapse()
p0_num = gather_value(evaluate_pressure_at_point(p_h, x_num, y_num))

if rank == 0:
    time_values.append(0.0)
    p_num_values.append(p0_num)

# ============================================================
# Time loop
# ============================================================
# Solve the coupled problem at each time step and extract numerical pressure at the selected point.
t = 0.0
for n in range(num_steps):
    t += dt
    num_its, converged = solver.solve(X0)
    X0.x.scatter_forward()
    Xn.x.array[:] = X0.x.array
    Xn.x.scatter_forward()

    u_h, p_h = X0.split()[0].collapse(), X0.split()[1].collapse()
    file_u.write_function(u_h, t)
    file_p.write_function(p_h, t)

    p_num = gather_value(evaluate_pressure_at_point(p_h, x_num, y_num))
    if rank == 0:
        time_values.append(t)
        p_num_values.append(p_num)
        if (n + 1) % 50 == 0: print(f"Step {n + 1:4d}/{num_steps}, t = {t:.4f} s, Newton iterations = {num_its}, p_num(12.5,0) = {p_num:.6f} Pa")

file_u.close()
file_p.close()

# ============================================================
# Analytical comparison and plot
# ============================================================
# Compute analytical pressure, save numerical/analytical data, and generate validation figures.
if rank == 0:
    time_values = np.array(time_values, dtype=float)
    p_num_values = np.array(p_num_values, dtype=float)
    p_ana_values = mandel_pressure(time_values, x_analytical, a_mandel, beta_ns, pinit, c_value)

    df = pd.DataFrame({"Time_s": time_values, "Numerical_pressure_at_x_12p5_y_0_Pa": p_num_values, "Analytical_pressure_at_xprime_2p5_Pa": p_ana_values})
    csv_path = os.path.join(results_dir, "mandel_numerical_vs_analytical_pressure_x12p5_y0.csv")
    df.to_csv(csv_path, index=False)

    print(f"Initial numerical pressure  = {p_num_values[0]:.6f} Pa")
    print(f"Initial analytical pressure = {p_ana_values[0]:.6f} Pa")
    print(f"Final numerical pressure    = {p_num_values[-1]:.6f} Pa")
    print(f"Final analytical pressure   = {p_ana_values[-1]:.6f} Pa")

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(time_values, p_ana_values, "-", linewidth=2.2, label="Analytical Mandel")
    plt.plot(time_values, p_num_values, "--", linewidth=2.2, label="Numerical FEniCSx")
    plt.xlabel("Time, t (s)")
    plt.ylabel("Excess pore pressure, p (Pa)")
    plt.title("Pore pressure comparison at numerical point (12.5, 0)")
    plt.grid(True, alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()

    png_path = os.path.join(results_dir, "mandel_pressure_comparison_x12p5_y0.png")
    pdf_path = os.path.join(results_dir, "mandel_pressure_comparison_x12p5_y0.pdf")
    svg_path = os.path.join(results_dir, "mandel_pressure_comparison_x12p5_y0.svg")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.show()

    print(f"CSV saved to     : {os.path.abspath(csv_path)}")
    print(f"PNG figure saved : {os.path.abspath(png_path)}")
    print(f"PDF figure saved : {os.path.abspath(pdf_path)}")
    print(f"SVG figure saved : {os.path.abspath(svg_path)}")
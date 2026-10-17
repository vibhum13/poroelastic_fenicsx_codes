#!/usr/bin/env python3
"""10 Hz Lo (2010) harmonic-loading problem: fully dynamic u-p-w FEM.

A saturated poroelastic half-space is represented by a narrow 2D strip.
The top is drained and carries -P0*abs(cos(2*pi*10*t)).
Initially, p_f=P0 and sigma_zz=-P0, with zero velocity and fluid flux.
The bottom is truncated with an approximate outgoing-wave impedance.
CG1-DG0-BDM1 elements solve the displacement-pressure-flux system.
Backward Euler is used for first-order rates, and Newmark for acceleration.

Run: mpirun -n 4 python3 Lo_UPQ_FEM_10Hz_fast.py
Test: mpirun -n 4 python3 Lo_UPQ_FEM_10Hz_fast.py --quick
"""

import argparse
import csv
import time
from pathlib import Path
import numpy as np

# Lo (2010) material properties, SI units. Symbols follow the manuscript.
Kd, Ks, Kf, mu = 8.33e6, 35.0e9, 2.25e9, 3.85e6
rho_s, rho_f, phi0 = 2650.0, 997.0, 0.45
kappa, mu_f, tor = 5.3e-13, 1.0e-3, 1.0
ab = 1.0 - Kd/Ks
M = 1.0/((ab-phi0)/Ks + phi0/Kf)
Ku = Kd + ab**2*M
rho = (1.0-phi0)*rho_s + phi0*rho_f
rho_rel = tor*rho_f/phi0
R = mu_f/kappa
H = Ku + 4.0*mu/3.0                    # Uniaxial undrained stiffness
c1 = np.sqrt(H/rho)
c2 = np.sqrt((M*H - (ab*M)**2)/(rho_rel*H - rho_f*ab*M))
DL = R*H/(rho_rel*H - rho_f*ab*M)
cv = c2**2/DL


def options():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true', help='0.2 s diagnostic; NOT the 60 s result')
    ap.add_argument('--time', type=float, default=60.0)
    ap.add_argument('--steps-per-cycle', type=int, default=100)
    ap.add_argument('--domain-depth', type=float, default=200.0)
    ap.add_argument('--depth-plotted', type=float, default=100.0)
    ap.add_argument('--pressure', type=float, default=1000.0)
    ap.add_argument('--nx', type=int, default=2)
    ap.add_argument('--vtk-interval', type=float, default=10.0)
    ap.add_argument('--sample-interval', type=float, default=0.25)
    ap.add_argument('--out', default='LO_2010_REPRODUCTION_CAUSAL_UPQ_NEW')
    ap.add_argument('--boundary-check', action='store_true')
    x = ap.parse_args()
    if x.quick:
        x.time = min(x.time, 0.2)
        x.steps_per_cycle = 48
    if x.domain_depth <= x.depth_plotted or x.nx < 2 or x.time <= 0:
        ap.error('Check computational depth, observation depth, mesh width and final time.')
    if x.steps_per_cycle < 24:
        ap.error('At least 24 steps per cosine period are required.')
    return x


def save_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)


def save_excel(path, tables):
    """Small .xlsx writer using the standard library (no extra installations)."""
    from zipfile import ZipFile, ZIP_DEFLATED
    from xml.etree import ElementTree as ET

    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    pkg = 'http://schemas.openxmlformats.org/package/2006/relationships'
    typ = 'http://schemas.openxmlformats.org/package/2006/content-types'
    sheets = [(k, v) for k, v in tables.items() if v]
    ET.register_namespace('', ns)
    ET.register_namespace('r', rel)

    def xml(elem):
        return ET.tostring(elem, encoding='utf-8', xml_declaration=True)

    def col_name(j):
        s = ''
        while j:
            j, r = divmod(j-1, 26)
            s = chr(65+r)+s
        return s

    book = ET.Element(f'{{{ns}}}workbook')
    book_sheets = ET.SubElement(book, f'{{{ns}}}sheets')
    book_rel = ET.Element(f'{{{pkg}}}Relationships')
    root_rel = ET.Element(f'{{{pkg}}}Relationships')
    kinds = ET.Element(f'{{{typ}}}Types')
    ET.SubElement(kinds, f'{{{typ}}}Default', Extension='rels', ContentType='application/vnd.openxmlformats-package.relationships+xml')
    ET.SubElement(kinds, f'{{{typ}}}Default', Extension='xml', ContentType='application/xml')
    ET.SubElement(kinds, f'{{{typ}}}Override', PartName='/xl/workbook.xml', ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml')
    ET.SubElement(root_rel, f'{{{pkg}}}Relationship', Id='rId1', Type=rel+'/officeDocument', Target='xl/workbook.xml')

    with ZipFile(path, 'w', compression=ZIP_DEFLATED) as zf:
        for i, (name, rows) in enumerate(sheets, 1):
            ET.SubElement(book_sheets, f'{{{ns}}}sheet', {'name': name, 'sheetId': str(i), f'{{{rel}}}id': f'rId{i}'})
            ET.SubElement(book_rel, f'{{{pkg}}}Relationship', Id=f'rId{i}', Type=rel+'/worksheet', Target=f'worksheets/sheet{i}.xml')
            ET.SubElement(kinds, f'{{{typ}}}Override', PartName=f'/xl/worksheets/sheet{i}.xml', ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml')
            sheet = ET.Element(f'{{{ns}}}worksheet')
            view = ET.SubElement(ET.SubElement(sheet, f'{{{ns}}}sheetViews'), f'{{{ns}}}sheetView', workbookViewId='0')
            ET.SubElement(view, f'{{{ns}}}pane', ySplit='1', topLeftCell='A2', activePane='bottomLeft', state='frozen')
            data = ET.SubElement(sheet, f'{{{ns}}}sheetData')
            columns = list(rows[0])
            for k, row in enumerate([dict(zip(columns, columns))] + rows, 1):
                row_node = ET.SubElement(data, f'{{{ns}}}row', r=str(k))
                for j, field in enumerate(columns, 1):
                    value = row[field]
                    cell = ET.SubElement(row_node, f'{{{ns}}}c', r=f'{col_name(j)}{k}')
                    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                        ET.SubElement(cell, f'{{{ns}}}v').text = str(value)
                    else:
                        cell.set('t', 'inlineStr')
                        ET.SubElement(ET.SubElement(cell, f'{{{ns}}}is'), f'{{{ns}}}t').text = str(value)
            zf.writestr(f'xl/worksheets/sheet{i}.xml', xml(sheet))
        zf.writestr('[Content_Types].xml', xml(kinds))
        zf.writestr('_rels/.rels', xml(root_rel))
        zf.writestr('xl/workbook.xml', xml(book))
        zf.writestr('xl/_rels/workbook.xml.rels', xml(book_rel))


def make_mesh(comm, args, f):
    import basix.ufl
    import ufl
    from dolfinx import mesh

    L = args.domain_depth
    skin = min(0.15, max(0.012, 5.0*np.sqrt(cv/(np.pi*f))))
    z1 = np.linspace(0.0, skin, 91)[:-1]
    z2 = np.linspace(skin, 0.5, 66)[:-1]
    z3 = np.linspace(0.5, 10.0, 111)[:-1]
    nz = max(240, int(np.ceil((L-10.0)/(c1/(2.0*f*24.0)))))
    if args.quick:
        nz = max(160, int(np.ceil((L-10.0)/(c1/(2.0*f*15.0)))))
    z4 = np.linspace(10.0, L, nz+1)
    z = np.r_[z1, z2, z3, z4]/L
    x = np.linspace(0.0, 0.06, args.nx+1)

    # The global mesh is built on rank 0 and partitioned by DOLFINx.
    if comm.rank == 0:
        pts = np.array([[xi, zi] for zi in z for xi in x])
        cells = []
        for i in range(len(z)-1):
            for j in range(args.nx):
                a = i*(args.nx+1)+j
                cells.extend(((a, a+1, a+args.nx+2), (a, a+args.nx+2, a+args.nx+1)))
        cells = np.asarray(cells, np.int64)
    else:
        pts, cells = np.empty((0, 2)), np.empty((0, 3), np.int64)

    element = basix.ufl.element('Lagrange', 'triangle', 1, shape=(2,))
    msh = mesh.create_mesh(comm, cells, pts, ufl.Mesh(element),
                           partitioner=mesh.create_cell_partitioner(mesh.GhostMode.shared_facet))
    return msh, len(z)-1


def solve_fem(args, folder):
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import fem, mesh, geometry, io, log
    from dolfinx.fem import petsc as fp
    import basix.ufl
    import ufl

    comm = MPI.COMM_WORLD
    f = 10.0
    L, tend, P0 = args.domain_depth, args.time, args.pressure
    tc = L/c1
    nt = int(np.ceil(args.steps_per_cycle*f*tend))
    dt_s, dt0 = tend/nt, (tend/nt)/tc

    def message(s):
        if comm.rank == 0:
            print(s, flush=True)

    message(f'10 Hz | {comm.size} MPI ranks | t={tend:g} s | {nt} steps | dt={dt_s:g} s')
    msh, nz = make_mesh(comm, args, f)
    message(f'Mesh ready: {nz} cells vertically, {args.nx} horizontally')
    dim = msh.topology.dim-1
    msh.topology.create_entities(dim)
    faces = lambda test: mesh.locate_entities_boundary(msh, dim, test)
    top = faces(lambda x: np.isclose(x[1], 0.0))
    bottom = faces(lambda x: np.isclose(x[1], 1.0))
    left = faces(lambda x: np.isclose(x[0], 0.0))
    right = faces(lambda x: np.isclose(x[0], 0.06))
    facets = np.r_[top, bottom, left, right].astype(np.int32)
    ids = np.r_[np.full(len(top), 1), np.full(len(bottom), 2),
                np.full(len(left), 3), np.full(len(right), 4)].astype(np.int32)
    order = np.argsort(facets)
    tags = mesh.meshtags(msh, dim, facets[order], ids[order])
    dx = ufl.Measure('dx', domain=msh)
    ds = ufl.Measure('ds', domain=msh, subdomain_data=tags)

    eu = basix.ufl.element('Lagrange', 'triangle', 1, shape=(2,))
    ep = basix.ufl.element('DG', 'triangle', 0)
    ew = basix.ufl.element('BDM', 'triangle', 1)
    W = fem.functionspace(msh, basix.ufl.mixed_element([eu, ep, ew]))
    Vu, map_u = W.sub(0).collapse()
    Vp, map_p = W.sub(1).collapse()
    Vw, map_w = W.sub(2).collapse()
    sol, old = fem.Function(W, name='u_p_w'), fem.Function(W, name='u_p_w_old')
    vel, acc = fem.Function(Vu), fem.Function(Vu)

    # Initial pressure and stress are p/P0=1 and sigma_zz/P0=-1.
    u0, p0, w0 = fem.Function(Vu), fem.Function(Vp), fem.Function(Vw)
    eps0 = (ab-1.0)*H/(Kd+4.0*mu/3.0)
    u0.interpolate(lambda x: np.vstack((np.zeros(x.shape[1]), eps0*x[1])))
    p0.x.array[:] = 1.0
    w0.x.array[:] = 0.0
    for field in (u0, p0, w0, vel, acc):
        field.x.scatter_forward()
    map_u, map_p, map_w = (np.asarray(m, np.int32) for m in (map_u, map_p, map_w))
    old.x.array[map_u], old.x.array[map_p], old.x.array[map_w] = u0.x.array, p0.x.array, w0.x.array
    old.x.scatter_forward()
    sol.x.array[:] = old.x.array
    sol.x.scatter_forward()

    # Impermeable sides/bottom and zero horizontal displacement.
    walls = np.unique(np.r_[bottom, left, right]).astype(np.int32)
    udofs = fem.locate_dofs_topological(W.sub(0).sub(0), dim, walls)
    wdofs = fem.locate_dofs_topological((W.sub(2), Vw), dim, walls)
    bcs = [fem.dirichletbc(PETSc.ScalarType(0), udofs, W.sub(0).sub(0)),
           fem.dirichletbc(fem.Function(Vw), wdofs, W.sub(2))]

    # The problem is linear: build the matrix once, then update only the RHS.
    u, p, w = ufl.TrialFunctions(W)
    un, pn, wn = ufl.split(old)
    vu, vp, vw = ufl.TestFunctions(W)
    dt = fem.Constant(msh, PETSc.ScalarType(dt0))
    load = fem.Constant(msh, PETSc.ScalarType(1.0))
    beta, gamma = 0.25, 0.5
    anew = (u-un-dt*vel-(0.5-beta)*dt**2*acc)/(beta*dt**2)
    vnew = vel+dt*((1.0-gamma)*acc+gamma*anew)
    udot, pdot, wdot = (u-un)/dt, (p-pn)/dt, (w-wn)/dt
    la = (Kd-2.0*mu/3.0)/H
    sh = mu/H
    rf, rr = rho_f/rho, rho_rel/rho
    Rh, St = R*L**2/(H*tc), H/M

    def sigma(uu, pp):
        return la*ufl.div(uu)*ufl.Identity(2) + 2.0*sh*ufl.sym(ufl.grad(uu)) - ab*pp*ufl.Identity(2)

    F = (St*pdot+ab*ufl.div(udot)+ufl.div(w))*vp*dx
    F += (ufl.inner(sigma(u,p),ufl.grad(vu))+ufl.dot(anew,vu)+rf*ufl.dot(wdot,vu))*dx
    F -= load*vu[1]*ds(1)
    F += (1.0+vnew[1])*vu[1]*ds(2)    # Approximate outgoing fast-wave condition
    F += (Rh*ufl.dot(w,vw)+rf*ufl.dot(anew,vw)+rr*ufl.dot(wdot,vw)-p*ufl.div(vw))*dx
    # Natural top condition from the flux weak form: p_f=0.
    a = fem.form(ufl.lhs(F))
    rhs = fem.form(ufl.rhs(F))
    A = fp.assemble_matrix(a, bcs=bcs)
    A.assemble()
    b = fp.create_vector(rhs)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType('mumps')
    ksp.setUp()
    log.set_log_level(log.LogLevel.WARNING)
    message('Constant system matrix assembled; MUMPS factorization will be reused')

    # Save numerical displacement, pressure, stress and relative fluid flux.
    uv = fem.Function(Vu, name='u_normalized')
    pv = fem.Function(Vp, name='p_over_P0')
    sv = fem.Function(Vp, name='sigma_zz_over_P0')
    Vq = fem.functionspace(msh, ('Lagrange', 1, (2,)))
    qv = fem.Function(Vq, name='relative_flux_scaled')
    ue = fem.Expression(ufl.split(sol)[0], Vu.element.interpolation_points())
    pe = fem.Expression(ufl.split(sol)[1], Vp.element.interpolation_points())
    se = fem.Expression(sigma(*ufl.split(sol)[:2])[1,1], Vp.element.interpolation_points())
    we = fem.Expression(ufl.split(sol)[2], Vq.element.interpolation_points())

    def update_fields():
        uv.interpolate(ue); pv.interpolate(pe); sv.interpolate(se); qv.interpolate(we)
        for v in (uv,pv,sv,qv):
            v.x.scatter_forward()

    vtkdir = folder/'VTK'
    if comm.rank == 0:
        vtkdir.mkdir(parents=True, exist_ok=True)
    comm.barrier()
    vtk = {name: io.VTKFile(comm, str(vtkdir/f'{name}.pvd'), 'w') for name in ('u','p','q','sigma')}

    def write_vtk(t):
        update_fields()
        for name, field in (('u',uv),('p',pv),('q',qv),('sigma',sv)):
            vtk[name].write_function(field,t)

    # Each rank evaluates its own cells; MPI sums all collected values.
    owned = np.arange(msh.topology.index_map(msh.topology.dim).size_local, dtype=np.int32)
    tree = geometry.bb_tree(msh,msh.topology.dim,owned)

    def locations(z):
        xyz = np.c_[np.full(len(z),0.03),z,np.zeros(len(z))]
        collisions = geometry.compute_collisions_points(tree,xyz)
        cells = geometry.compute_colliding_cells(msh,collisions,xyz)
        ii = np.array([j for j in range(len(z)) if len(cells.links(j))],np.int32)
        cc = np.array([cells.links(j)[0] for j in ii],np.int32)
        return xyz[ii],cc,ii,len(z)

    def read_values(points):
        xyz, cells, ii, n = points
        local = np.zeros((n,3))
        if len(ii):
            local[ii,0] = pv.eval(xyz,cells).reshape(-1)
            local[ii,1] = sv.eval(xyz,cells).reshape(-1)
            local[ii,2] = 1.0
        total = np.zeros_like(local)
        comm.Allreduce(local,total,op=MPI.SUM)
        if np.any(total[:,2] == 0):
            raise RuntimeError('Some sample points were not found in the mesh.')
        return total[:,:2]/total[:,2,None]

    depths = np.unique(np.r_[0.001,np.linspace(0.005,1,65),np.linspace(1.1,10,55),
                              np.linspace(10.4,args.depth_plotted-0.001,260)])
    sensor_z = np.array([z for z in (1.0,20.0,60.0,100.0) if z < L])
    profile_pts, sensor_pts = locations(depths/L), locations(sensor_z/L)
    histories = []
    vtk_steps = max(1,round(args.vtk_interval/dt_s))
    sample_steps = max(1,round(args.sample_interval/dt_s))
    early = {max(1,round(t/dt_s)) for t in (0.02,0.1,0.5,1.0,5.0) if t<tend}
    checkpoints = {1, 10, 50, 100, 250, 500, 1000, nt} | set(range(2000, nt+1, 2000))
    tic = time.monotonic()
    write_vtk(0.0)
    message('Starting time stepping')

    for step in range(1,nt+1):
        t = step*dt_s
        load.value = PETSc.ScalarType(abs(np.cos(2.0*np.pi*f*t)))
        with b.localForm() as loc:
            loc.set(0.0)
        fp.assemble_vector(b,rhs)
        fp.apply_lifting(b,[a],bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,mode=PETSc.ScatterMode.REVERSE)
        fp.set_bc(b,bcs)
        ksp.solve(b,sol.x.petsc_vec)
        if ksp.getConvergedReason() <= 0:
            raise RuntimeError(f'Linear solve failed at step {step}, PETSc reason={ksp.getConvergedReason()}')
        sol.x.scatter_forward()

        # Keep the same Newmark update as the original working code.
        du = sol.x.array[map_u]-old.x.array[map_u]
        an = (du-dt0*vel.x.array-(0.5-beta)*dt0**2*acc.x.array)/(beta*dt0**2)
        vn = vel.x.array+dt0*((1.0-gamma)*acc.x.array+gamma*an)
        acc.x.array[:] = an; vel.x.array[:] = vn
        acc.x.scatter_forward(); vel.x.scatter_forward()
        old.x.array[:] = sol.x.array; old.x.scatter_forward()

        if step%sample_steps == 0 or step > nt-2*args.steps_per_cycle or step == nt:
            update_fields()
            values = read_values(sensor_pts)
            if comm.rank == 0:
                for z,(pp,ss) in zip(sensor_z,values):
                    histories.append({'frequency_Hz':f,'time_s':t,'depth_m':float(z),
                                      'p_numerical':float(pp),'sigma_numerical':float(ss)})
        if step in early or step%vtk_steps == 0 or step == nt:
            write_vtk(t)
        if step in checkpoints:
            message(f'{100*step/nt:5.1f}%  step {step}/{nt}  t={t:.3f}s  elapsed={(time.monotonic()-tic)/60:.1f} min')

    for file in vtk.values():
        file.close()
    comm.barrier()
    update_fields()
    final = read_values(profile_pts)
    ksp.destroy(); A.destroy(); b.destroy()
    if comm.rank != 0:
        return []

    profiles = [{'frequency_Hz':f,'time_s':tend,'depth_m':float(z),
                 'p_numerical':float(v[0]),'sigma_numerical':float(v[1])}
                for z,v in zip(depths,final)]
    params = [{'parameter':k,'value':v} for k,v in (
        ('Benchmark','Lo et al. (2010), harmonic half-space'),
        ('FEM','FEniCSx 0.9 CG1-DG0-BDM1 dynamic u-p-w'),
        ('Integration','Backward Euler; Newmark beta=0.25 gamma=0.5'),
        ('Frequency Hz',f),('Time s',tend),('Time step s',dt_s),
        ('Domain depth m',L),('Observed depth m',args.depth_plotted),
        ('Vertical cells',nz),('P0 Pa',P0),('Tortuosity',tor),
        ('Bottom boundary','Approximate outgoing impedance; impermeable'),
        ('c1 m/s',c1),('c2 m/s',c2),('DL 1/s',DL))]
    tables = {'Parameters':params,'Depth_profiles':profiles,'Time_histories':histories}
    for name,rows in tables.items():
        save_csv(folder/f'{name}.csv',rows)
    save_excel(folder/'Lo_2010_mixed_UPQ_validation.xlsx',tables)
    message('CSV, Excel and VTK saved')
    return profiles


def make_plots(profiles, out, t):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':18,
                         'axes.labelsize':24,'xtick.labelsize':19,'ytick.labelsize':19})
    folder = out/'PAPER_FIGURES_ONLY'
    folder.mkdir(parents=True, exist_ok=True)
    depth = np.array([r['depth_m'] for r in profiles])
    for key,col,label,num in (('p','p_numerical',r'Normalized pore pressure, $p/P_0$',1),
                              ('sigma','sigma_numerical',r'Normalized total vertical stress, $\sigma_{zz}/P_0$',4)):
        fig,ax = plt.subplots(figsize=(8.9,4.6))
        ax.plot([r[col] for r in profiles],depth,'-',lw=2.2,label='Numerical (FEniCSx UPQ)')
        ax.set(xlabel=label,ylabel='Depth (m)',title=f'Lo Fig. {num}:  $f=10$ Hz,  $t={t:g}$ s')
        ax.set_ylim(float(np.max(depth)),0.0)
        ax.set_xlim(-0.25,1.10) if key=='p' else ax.set_xlim(-1.1,0.25)
        ax.minorticks_on()
        ax.grid(True,ls=':',alpha=0.45,which='major')
        ax.grid(True,ls=':',alpha=0.18,which='minor')
        ax.legend(frameon=False,fontsize=13,loc='best')
        fig.tight_layout()
        stem = folder/f'Lo_Fig{num}_{key}_FEM_only'
        fig.savefig(str(stem)+'.png',dpi=300,bbox_inches='tight')
        fig.savefig(str(stem)+'.pdf',bbox_inches='tight')
        plt.close(fig)


def main():
    from mpi4py import MPI
    args = options()
    comm = MPI.COMM_WORLD
    out = Path(args.out)/f'Fig1_t{args.time:g}s'
    folder = out/'f_10Hz'
    if comm.rank == 0:
        folder.mkdir(parents=True,exist_ok=True)
    comm.barrier()
    profiles = solve_fem(args,folder)

    if args.boundary_check:
        old_depth = args.domain_depth
        try:
            args.domain_depth = 1.5*old_depth
            other = solve_fem(args,folder/'boundary_check_deeper_domain')
        finally:
            args.domain_depth = old_depth
        if comm.rank == 0:
            z = np.array([r['depth_m'] for r in profiles])
            diff = np.array([r['p_numerical']-s['p_numerical'] for r,s in zip(profiles,other)])
            ref = np.array([s['p_numerical'] for s in other])
            err = 100*np.sqrt(np.trapz(diff**2,z)/max(np.trapz(ref**2,z),1e-20))
            save_csv(folder/'Boundary_reflection_sensitivity.csv',
                     [{'frequency_Hz':10.,'original_depth_m':old_depth,'deeper_depth_m':1.5*old_depth,
                       'relative_L2_pressure_difference_percent':float(err)}])
    if comm.rank == 0:
        make_plots(profiles,out,args.time)
        print('10 Hz FEM complete. Output:',out,flush=True)


if __name__ == '__main__':
    main()

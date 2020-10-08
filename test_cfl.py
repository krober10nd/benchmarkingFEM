from firedrake import *
set_log_level(ERROR)
import numpy as np

from benchmarker import solver_CG, _build_space

## defining method to analyze
space = "spectral"
cell_type = "quad"
error_threshold = 1e-1

if cell_type == "quad":
    quadrilateral = True
elif cell_type == "tria":
    quadrilateral = False
else:
    raise ValueError("Cell type not supported")


## building reference solution
p = 4
timestep = 0.0005
mesh = UnitSquareMesh(50, 50, quadrilateral = quadrilateral)
ref = solver_CG(mesh, el=cell_type, space=space, deg=p, T=0.50, dt=timestep)
V_fine = _build_space(mesh,cell_type, space,p)

## checking dt as f(p) in mesh
degrees = [1,2,3,4]
ns = [40, 50, 60, 70]
dts = np.zeros((len(ns),len(degrees)))
print(dts)
ref_dt = 0.0001

i = 0
for n in ns:
    mesh = UnitSquareMesh(n, n, quadrilateral = quadrilateral)
    ### building reference solution
    ref = solver_CG(mesh, el=cell_type, space=space, deg=p, T=0.50, dt=ref_dt)


    j = 0
    for degree in degrees:
        error = 1e-10 # value to enter while loop
        dt = 0.0005
        while error < error_threshold:
            dt += dt
            try:
                sol = solver_CG(mesh, el=cell_type, space=space, deg=degree, T=0.50, dt=dt)
                error = errornorm(ref, sol)
            except:
                error = 1e10

        dt = dt/2

        error = 1e-10 # value to enter while loop
        while error < error_threshold:
            dt += dt/4.
            try:
                sol = solver_CG(mesh, el=cell_type, space=space, deg=degree, T=0.50, dt=dt)
                error = errornorm(ref, sol)
            except:
                error = 1e10
        dt = 4./5.*dt

        error = 1e-10 # value to enter while loop
        while error < error_threshold:
            dt += dt/10.
            try:
                sol = solver_CG(mesh, el=cell_type, space=space, deg=degree, T=0.50, dt=dt)
                error = errornorm(ref, sol)
            except:
                error = 1e10

        dt = dt*9./10.

        dts[i,j] = dt
        j+=1
    i+=1


print(dts)





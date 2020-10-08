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
degrees = np.array([1,2,3,4])
ns = np.array([50, 100, 200, 400, 800])
dts = np.zeros((len(ns),len(degrees)))
print(dts)
ref_dts = [0.0005, 0.0003, 0.0003, 6e-5]

i = 0
for n in ns:
    mesh = UnitSquareMesh(n, n, quadrilateral = quadrilateral)
    ### building reference solution
    ref_dt = ref_dts[i]
    ref = solver_CG(mesh, el=cell_type, space=space, deg=p, T=0.50, dt=ref_dt)
    j = 0
    for degree in degrees:
        dt = 0.0005
        for two_pot in [-1,0,1,2,3,4]:
            error = 1e-10 # value to enter while loop
            while error < error_threshold:
                incr = dt/(2**two_pot)
                dt += incr
                try:
                    sol = solver_CG(mesh, el=cell_type, space=space, deg=degree, T=0.50, dt=dt)
                    error = errornorm(ref, sol)
                except:
                    error = 1e10
            dt-= incr

        dts[i,j] = dt
        j+=1
    i+=1

print(dts) 


def find_fp(ns, degrees,dts):
    xs = 1./ns



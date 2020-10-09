from firedrake import *
set_log_level(ERROR)
import numpy as np

from benchmarker import solver_CG, _build_space


p = 2
timestep = 0.0005
space = "spectral"
cell_type = "quad"

if cell_type == "quad":
    quadrilateral = True
elif cell_type == "tria":
    quadrilateral = False
else:
    raise ValueError("Cell type not supported")


mesh = UnitSquareMesh(50, 50, quadrilateral = quadrilateral)
mh = MeshHierarchy(mesh, 2)
sols = [
    solver_CG(mesh, el=cell_type, space=space, deg=p, T=0.50, dt=timestep) for mesh in mh
]
coarse_on_fine = []
V = _build_space(mh[-1],cell_type, space,p) 
for sol in sols[:-1]:
    fine = Function(V).assign(sols[-1])
    prolong(sol, fine)
    coarse_on_fine.append(fine)
errors = [
    errornorm(coarse_on_fine[0], coarse_on_fine[1]),
    errornorm(coarse_on_fine[1], sols[-1]),
]
errors = np.asarray(errors)
l2conv = np.log2(errors[:-1] / errors[1:])
print(errors)
print(l2conv)

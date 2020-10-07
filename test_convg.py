from firedrake import *

import numpy as np

from benchmarker import solver_CG


p = 2
timestep = 0.0005
space = "KMV"


mesh = UnitSquareMesh(50, 50)
mh = MeshHierarchy(mesh, 2)
sols = [
    solver_CG(mesh, el="tria", space=space, deg=p, T=0.50, dt=timestep) for mesh in mh
]
coarse_on_fine = []
V = FunctionSpace(mh[-1], space, p)
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

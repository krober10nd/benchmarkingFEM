from firedrake import *

import numpy as np

from benchmarker import solver_CG


p = 1
timestep = 0.0005


mesh = UnitSquareMesh(50, 50)
mh = MeshHierarchy(mesh, 2)
print(len(mh))
sols = [
    solver_CG(mesh, el="tria", space="CG", deg=p, T=0.50, dt=timestep) for mesh in mh
]
coarse_on_fine = []
V = FunctionSpace(mh[-1], "CG", p)
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

from firedrake import *

import numpy as np

from benchmarker import solver_CG

mesh = UnitSquareMesh(10, 10)
mh = MeshHierarchy(mesh, 3)
sols = [solver_CG(mesh, el="tria", space="CG", deg=1, T=0.50, dt=0.001) for mesh in mh]
coarse_on_fine = [prolong(sol, sols[-1]) for sol in sols[:-1]]
errors = [
    errornorm(coarse_on_fine[0], coarse_on_fine[1]),
    errornorm(coarse_on_fine[1], coarse_on_fine[2]),
]
errors = np.asarray(errors)
l2conv = np.log2(errors[:-1] / errors[1:])
print(l2conv)

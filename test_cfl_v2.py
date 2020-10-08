from firedrake import (
    errornorm,
    UnitSquareMesh,
    Function,
    FunctionSpace,
    set_log_level,
    ERROR,
    prolong,
    MeshHierarchy,
)

import numpy as np

from benchmarker import solver_CG, _build_space

set_log_level(ERROR)

# defining method to analyze
space = "CG"
cell_type = "tria"
error_threshold = 1e-3

if cell_type == "quad":
    quadrilateral = True
elif cell_type == "tria":
    quadrilateral = False
else:
    raise ValueError("Cell type not supported")


# building reference solution
p = 1
initial_dt = 0.0005

# determining dt as f(p) in mesh
degrees = [1, 2, 3, 4]
mesh = UnitSquareMesh(40, 40, quadrilateral=quadrilateral)
mh = MeshHierarchy(mesh, 3)

# determine maximum stable timestep with error < error_threshold
dts = np.zeros((len(mh), len(degrees)))

# construct reference solution
ref = solver_CG(mh[-1], el=cell_type, space=space, deg=p, T=0.50, dt=initial_dt)
V_fine = _build_space(mh[-1], cell_type, space, p)

# step to increase timestep (step << dt)
step = initial_dt / 10

for i, mesh in enumerate(mh[:-1]):

    for j, degree in enumerate(degrees):

        V_coarse = FunctionSpace(mesh, space, degree)

        error = 0.0
        dt = initial_dt

        for two_pot in [-1,0,1,2,3,4]:
            while error < error_threshold:
                # error < error_threshold so increase dt by small amount
                step = dt/(2**two_pot)
                dt = dt + step
                # NB: prolong does not copy so we must re-assign ref sol.
                fine = Function(V_fine).assign(ref)
                # Run the simulation with this dt
                try:
                    sol = solver_CG(
                        mesh, el=cell_type, space=space, deg=degree, T=0.50, dt=dt
                    )
                    error = errornorm(ref, prolong(sol, fine))
                except Exception:
                    # numerical instability occurred, exit with last stable dt
                    dt = dt - step
                    error = 1e10
                #print(
                 #   "For degree {}, the error is {} using a {} s timestep".format(
                  #      degree, error, dt
                   # )
            #)
        # error > error_threshold,
        print(
            "Highest stable dt is {} s for a degree {} for a an error threshold of {}".format(
                dt, degree, error_threshold
            )
        )
        dts[i, j] = dt

print("------FINISHED------")
print(dts)

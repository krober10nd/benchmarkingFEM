# Benchmark computational performance of higher-order mass-lumped FEM
import firedrake as fd
from firedrake import Constant, dx, exp, dot, grad
import finat

import argparse
import math
import time

# import os


def RickerWavelet(t, freq=2, amp=1.0):
    """Time-varying source function"""
    # shift so full wavelet is developd
    t = t - 3 * (math.sqrt(6.0) / (math.pi * freq))
    return (
        amp
        * (1.0 - (1.0 / 2.0) * (2.0 * math.pi * freq) * (2.0 * math.pi * freq) * t * t)
        * math.exp(
            (-1.0 / 4.0) * (2.0 * math.pi * freq) * (2.0 * math.pi * freq) * t * t
        )
    )


def delta_expr(x0, x, z, y=None, sigma_x=2000.0):
    """Spatial function to apply source"""
    sigma_x = Constant(sigma_x)
    if y is None:
        return exp(-sigma_x * ((x - x0[0]) ** 2 + (z - x0[1]) ** 2))
    else:
        return exp(-sigma_x * ((x - x0[0]) ** 2 + (y - x0[1]) ** 2 + (z - x0[2]) ** 2))


def solver_CG(degree, T, dimension, dt=0.001, lump_mass=False):
    sd = dimension
    if sd == 2:
        mesh = fd.UnitSquareMesh(101, 201, diagonal="right")
    elif sd == 3:
        mesh = fd.UnitCubeMesh(20, 20, 20)

    if lump_mass:
        V = fd.FunctionSpace(mesh, "KMV", degree)
    else:
        V = fd.FunctionSpace(mesh, "CG", degree)

    if lump_mass:
        quad_rule = finat.quadrature.make_quadrature(
            V.finat_element.cell, V.ufl_element().degree(), "KMV"
        )

        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
        # if nspool > 0:
        #    outfile = File(os.getcwd() + "/results/simple_shots_lumped.pvd")
    else:
        quad_rule = None
        params = {"ksp_type": "cg", "pc_type": "jacobi"}
        # if nspool > 0:
        #    outfile = File(os.getcwd() + "/results/simple_shots.pvd")

    print("------------------------------------------")
    print("The problem has " + str(V.dof_dset.total_size) + " degrees of freedom.")
    print("------------------------------------------")

    nt = int(T / dt)  # number of timesteps

    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)

    u_np1 = fd.Function(V)  # n+1
    u_n = fd.Function(V)  # n
    u_nm1 = fd.Function(V)  # n-1

    # constant speed
    c = Constant(1.5)

    m = (
        (1.0 / (c * c))
        * (u - 2.0 * u_n + u_nm1)
        / Constant(dt * dt)
        * v
        * dx(rule=quad_rule)
    )  # mass-like matrix
    a = dot(grad(u_n), grad(v)) * dx  # stiffness matrix

    # injection of source into mesh
    if sd == 2:
        x, y = fd.SpatialCoordinate(mesh)
        source = Constant([0.5, 0.5])
        delta = fd.Interpolator(delta_expr(source, x, y), V)
    else:
        x, y, z = fd.SpatialCoordinate(mesh)
        source = Constant([0.5, 0.5, 0.5])
        delta = fd.Interpolator(delta_expr(source, x, y, z), V)

    t = 0.0

    ricker = Constant(0.0)
    f = fd.Function(delta.interpolate()) * ricker
    ricker.assign(RickerWavelet(t))

    F = m + a - f * v * dx  # form
    a = fd.lhs(F)
    r = fd.rhs(F)
    A = fd.assemble(a)
    R = fd.assemble(r)
    solver = fd.LinearSolver(A, P=None, solver_parameters=params)

    t1 = time.time()
    # timestepping loop
    for step in range(nt):

        t = step * float(dt)

        # update the source
        ricker.assign(RickerWavelet(t))

        solver.solve(u_np1, R)

        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        if step % 100 == 0:
            # outfile.write(u_n)
            print("Time is " + str(t), flush=True)
    return time.time() - t1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark scalar wave formulations..."
    )

    parser.add_argument(
        "--method",
        dest="method",
        type=str,
        default=False,
        required=True,
        help="Run benchmark with method=('cg','lump')",
    )

    parser.add_argument(
        "--degree",
        dest="degree",
        type=int,
        default=1,
        required=False,
        help="run benchmark with spatial degree=(degree)",
    )

    parser.add_argument(
        "--time",
        dest="T",
        type=float,
        default=1.0,
        required=False,
        help="Run benchmark for `T` simulation seconds",
    )

    parser.add_argument(
        "--dimension",
        dest="dimension",
        type=int,
        default=2,
        required=False,
        help="Run benchmark for in spatial dimension `sd`",
    )

    args = parser.parse_args()

    # Standard Lagrange FEM w or wo/ mass lumping
    if args.method == "lump" or args.method == "cg":
        if args.method == "lump":
            timing = solver_CG(args.degree, args.T, args.dimension, lump_mass=True)
            print("Wall-clock simulation time was: " + str(timing))
        else:
            timing = solver_CG(args.degree, args.T, args.dimension, lump_mass=False)
            print("Wall-clock simulation time was: " + str(timing))
    else:
        raise "unrecognized option"

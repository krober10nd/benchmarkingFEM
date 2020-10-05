# Benchmark computational performance of higher-order mass-lumped FEM
import firedrake as fd
from firedrake import Constant, dx, dot, grad, COMM_WORLD
from firedrake.petsc import PETSc
import finat
from mpi4py import MPI
import numpy as np

import time
import sys

from helpers import RickerWavelet, delta_expr

# import os

__all__ = ["solver_CG"]

PETSc.Log().begin()


def get_time(event, comm=COMM_WORLD):
    return (
        comm.allreduce(PETSc.Log.Event(event).getPerfInfo()["time"], op=MPI.SUM)
        / comm.size
    )


N = 20


def solver_CG(el, deg, sd, T, dt=0.001, lump_mass=False, warm_up=False):
    if sd == 2:
        if el == "tria":
            mesh = fd.UnitSquareMesh(N, N)
        elif el == "quad":
            mesh = fd.UnitSquareMesh(N, N, quadilateral=True)
        else:
            raise ValueError("Unrecognized element type")
    elif sd == 3:
        if el == "tetra":
            mesh = fd.UnitCubeMesh(N, N, N)
        elif el == "quad":
            mesh = fd.UnitCubeMesh(N, N, N, quadilateral=True)
        else:
            raise ValueError("Unrecognized element type")

    if lump_mass:
        V = fd.FunctionSpace(mesh, "KMV", deg)
    else:
        V = fd.FunctionSpace(mesh, "CG", deg)

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

    tot_dof = COMM_WORLD.allreduce(V.dof_dset.total_size, op=MPI.SUM)
    if COMM_WORLD.rank == 0:
        print("------------------------------------------")
        print("The problem has " + str(tot_dof) + " degrees of freedom.")
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
    solver = fd.LinearSolver(A, solver_parameters=params, options_prefix="")

    # timestepping loop
    results = []
    for step in range(nt):

        t = step * float(dt)

        # update the source
        ricker.assign(RickerWavelet(t))

        with PETSc.Log.Stage("{el}{deg}.N{N}".format(el=el, deg=deg, N=N)):
            solver.solve(u_np1, R)

            snes = get_time("SNESSolve")
            ksp = get_time("KSPSolve")
            pcsetup = get_time("PCSetUp")
            pcapply = get_time("PCApply")
            jac = get_time("SNESJacobianEval")
            residual = get_time("SNESFunctionEval")
            sparsity = get_time("CreateSparsity")

            results.append(
                [N, tot_dof, snes, ksp, pcsetup, pcapply, jac, residual, sparsity]
            )

        if warm_up:
            # Warm up symbolics/disk cache
            solver.solve(u_np1, R)
            sys.exit("Warming up...")

        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        if step % 100 == 0:
            # outfile.write(u_n)
            print("Time is " + str(t), flush=True)
    results = np.asarray(results)
    if mesh.comm.rank == 0:
        with open("scalar_wave.{el}.{deg}.csv".format(el=el, deg=deg), "w") as f:
            np.savetxt(
                f,
                results,
                fmt=["%d"] + ["%e"] * 8,
                delimiter=",",
                header="N,tot_dof,SNESSolve,KSPSolve,PCSetUp,PCApply,SNESJacobianEval,SNESFunctionEval,CreateSparsity",
                comments="",
            )


# Call the solvers to do the benchmarking!
solver_CG("tria", 1, 2, 1.0)

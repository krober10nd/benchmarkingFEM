# Benchmark computational performance of higher-order mass-lumped FEM
import firedrake as fd
from firedrake import Constant, dx, dot, grad, COMM_WORLD
from firedrake.petsc import PETSc
import finat
from mpi4py import MPI
import numpy as np

import sys

from helpers import RickerWavelet, delta_expr

import os

__all__ = ["solver_CG"]

PETSc.Log().begin()


def _get_time(event, comm=COMM_WORLD):
    return (
        comm.allreduce(PETSc.Log.Event(event).getPerfInfo()["time"], op=MPI.SUM)
        / comm.size
    )


def _get_mesh(el, sd, N):
    if sd == 2:
        if el == "tria":
            mesh = fd.UnitSquareMesh(N, N)
        elif el == "quad":
            mesh = fd.UnitSquareMesh(N, N, quadrilateral=True)
        else:
            raise ValueError("Unrecognized element type")
    elif sd == 3:
        if el == "tetra":
            mesh = fd.UnitCubeMesh(N, N, N)
        elif el == "hexa":
            mesh = fd.UnitCubeMesh(N, N, N, quadrilateral=True)
        else:
            raise ValueError("Unrecognized element type")
    else:
        raise ValueError("Invalid dimension")
    return mesh


def _get_space(mesh, el, deg, lump_mass):
    if lump_mass:
        if el == "tria":
            V = fd.FunctionSpace(mesh, "KMV", deg)
        elif el == "quad":
            V = fd.FunctionSpace(mesh, "CG", deg)
    else:
        V = fd.FunctionSpace(mesh, "CG", deg)

    return V


def _get_quad_rule(el, V, lump_mass):
    if lump_mass:
        if el == "tria":
            quad_rule = finat.quadrature.make_quadrature(
                V.finat_element.cell, V.ufl_element().degree(), "KMV"
            )
        elif el == "quad":
            quad_rule = finat.quadrature.make_quadrature(
                V.finat_element.cell, V.ufl_element().degree(), "gll"
            )
    else:
        quad_rule = None
    return quad_rule


def _select_params(lump_mass):
    if lump_mass:
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    else:
        params = {"ksp_type": "cg", "pc_type": "jacobi"}
    return params


def solver_CG(el, deg, sd, T, N=40, dt=0.001, lump_mass=False, warm_up=False):

    mesh = _get_mesh(el, sd, N)

    V = _get_space(mesh, el, deg, lump_mass)

    quad_rule = _get_quad_rule(el, V, lump_mass)

    params = _select_params(lump_mass)

    # DEBUG
    outfile = fd.File(os.getcwd() + "/results/simple_shots.pvd")

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
    ricker = Constant(0.0)
    source = Constant([0.5] * sd)
    coords = fd.SpatialCoordinate(mesh)
    F = m + a - delta_expr(source, *coords) * ricker * v * dx

    a, r = fd.lhs(F), fd.rhs(F)
    A, R = fd.assemble(a), fd.assemble(r)
    solver = fd.LinearSolver(A, solver_parameters=params, options_prefix="")

    # timestepping loop
    results = []

    t = 0.0
    for step in range(nt):

        with PETSc.Log.Stage("{el}{deg}.N{N}".format(el=el, deg=deg, N=N)):
            # update the source
            ricker.assign(RickerWavelet(t))

            solver.solve(u_np1, R)

            snes = _get_time("SNESSolve")
            ksp = _get_time("KSPSolve")
            pcsetup = _get_time("PCSetUp")
            pcapply = _get_time("PCApply")
            jac = _get_time("SNESJacobianEval")
            residual = _get_time("SNESFunctionEval")
            sparsity = _get_time("CreateSparsity")

            results.append(
                [N, tot_dof, snes, ksp, pcsetup, pcapply, jac, residual, sparsity]
            )

            u_nm1.assign(u_n)
            u_n.assign(u_np1)

            t = step * float(dt)

            if step % 10 == 0:
                outfile.write(u_n)
                print("Time is " + str(t), flush=True)

        if warm_up:
            # Warm up symbolics/disk cache
            solver.solve(u_np1, R)
            sys.exit("Warming up...")

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


# Call the solvers to do the benchmarking
solver_CG(el="tria", deg=1, sd=2, T=1.0, lump_mass=False)

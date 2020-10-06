# Benchmark computational performance of higher-order mass-lumped FEM
import firedrake as fd
from firedrake import Constant, dx, dot, grad, COMM_WORLD
from firedrake.petsc import PETSc
import finat
from mpi4py import MPI
import numpy as np

import sys

from helpers import RickerWavelet, delta_expr, gauss_lobatto_legendre_cube_rule

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
        if el == "tria":
            mesh = fd.UnitCubeMesh(N, N, N)
        elif el == "quad":
            mesh = fd.UnitCubeMesh(N, N, N, quadrilateral=True)
        else:
            raise ValueError("Unrecognized element type")
    else:
        raise ValueError("Invalid dimension")
    return mesh


def _build_space(mesh, el, space, deg):
    if el == "tria":
        V = fd.FunctionSpace(mesh, space, deg)
    elif el == "quad":
        if space == "spectral":
            element = fd.FiniteElement(
                "CG", mesh.ufl_cell(), degree=deg, variant="spectral"
            )
            V = fd.FunctionSpace(mesh, element)
        elif space == "S":
            V = fd.FunctionSpace(mesh, "S", deg)
        else:
            raise ValueError("Space not supported yet")
    return V


def _build_quad_rule(el, V, space):
    if el == "tria" and space == "KMV":
        quad_rule = finat.quadrature.make_quadrature(
            V.finat_element.cell,
            V.ufl_element().degree(),
            space,
        )
    elif el == "quad" and space == "spectral":
        quad_rule = gauss_lobatto_legendre_cube_rule(
            V.mesh().geometric_dimension(), V.ufl_element().degree()
        )
    elif el == "quad" and space == "S":
        quad_rule = None
    elif el == "tria" and space == "CG":
        quad_rule = None
    else:
        raise ValueError("Unsupported element/space combination")
    return quad_rule


def _select_params(space):
    if space == "KMV":
        params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    else:
        params = {"ksp_type": "cg", "pc_type": "jacobi"}
    return params


def solver_CG(el, space, deg, sd, T, N=40, dt=0.001, warm_up=False):
    """Solve the scalar wave equation on a unit square/cube using a
    CG FEM formulation with several different element types.

    Parameters
    ----------
    el: string
        The type of element either "tria" or "quad"
    space: string
        The space of the FEM. Available options are:
            "CG": Continuous Galerkin Finite Elements,
            "KMV": Kong-Mulder-Veldhuzien higher-order mass lumped elements
            "S" (for Serendipity) (NB: quad/hexs only)
            "spectral": tradiational spectral elements using GLL quad points (NB: quads/hexs only)
    deg: int
        The spatial polynomial degree.
    sd: int
        The geoemtric dimension of the problem.
    T: float
        The duration in simulation seconds of the simulation.
    N: int
        The number of grid points per dimension
    dt: float
        Simulation timestep
    warm_up: boolean, optional
        Warm up symbolics by running one timestep and shutting down.

    Returns
    -------
    None


    """

    mesh = _get_mesh(el, sd, N)

    V = _build_space(mesh, el, space, deg)

    quad_rule = _build_quad_rule(el, V, space)

    params = _select_params(space)

    # DEBUG
    outfile = fd.File(os.getcwd() + "/results/simple_shots.pvd")
    # END DEBUG

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
            ricker.assign(RickerWavelet(t, freq=6))

            R = fd.assemble(r, tensor=R)

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

        if warm_up:
            # Warm up symbolics/disk cache
            solver.solve(u_np1, R)
            sys.exit("Warming up...")

        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        t = step * float(dt)

        if step % 10 == 0:
            outfile.write(u_n)
            print("Time is " + str(t), flush=True)

    results = np.asarray(results)
    if mesh.comm.rank == 0:
        with open(
            "scalar_wave.{el}.{deg}.{space}.csv".format(el=el, deg=deg, space=space),
            "w",
        ) as f:
            np.savetxt(
                f,
                results,
                fmt=["%d"] + ["%e"] * 8,
                delimiter=",",
                header="N,tot_dof,SNESSolve,KSPSolve,PCSetUp,PCApply,SNESJacobianEval,SNESFunctionEval,CreateSparsity",
                comments="",
            )


# Test: call the solvers to do the benchmarking
solver_CG(el="quad", space="S", deg=2, sd=2, T=1.0, dt=0.001)

import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt

from util import (
    timeit,
    INFO,
    YAY,
    ERROR,
    WARN,
    diditwork,
)  # pylint: disable=unused-import

from pydrake.math import (
    le,
    ge,
    eq,
)  # pylint: disable=import-error, no-name-in-module, unused-import

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    SolverOptions,
    CommonSolverOption,
    IpoptSolver,
    SnoptSolver,
    GurobiSolver,
)
from pydrake.symbolic import (
    Polynomial,
    Variable,
    Variables,
    Expression,
)  # pylint: disable=import-error, no-name-in-module, unused-import

from program_options import ProgramOptions  # , FREE_POLY, PSD_POLY
from polynomial_gcs_dual import (
    PolynomialDualGCS,
    Vertex,
    Edge,
)  # pylint: disable=unused-import
from gcs_util import (
    make_quadratic_state_control_cost_function,
    make_quadratic_cost_function,
    get_vertex_name,
    get_mode_from_name,
)
from controls_graph import ControlsGraph, ControlsGraphEdge, ControlsGraphVertex
from polynomial_gcs_dual import Vertex


def control_policy(
    gcs: PolynomialDualGCS,
    graph: ControlsGraph,
    start_state: npt.NDArray,
    options: ProgramOptions,
) -> T.Tuple[npt.NDArray, npt.NDArray, float]:
    # figure out current set
    current_mode = None
    current_mode = options.system.get_mode_for_point(start_state)
    assert current_mode is not None, ERROR("start state inside not mode", start_state)

    horizon_index = 0
    current_vertex = gcs.vertices[get_vertex_name(horizon_index, current_mode)]
    current_state = start_state

    x_trajectory, u_trajectory, mode_trajectory = [start_state], [], [current_mode]

    total_cost = 0

    # find all edges out of current vertex
    edges_out = [
        gcs.edges[name] for name in current_vertex.edges_out
    ]  # type: T.List[Edge]
    results_out = []

    rolling_cost_function = make_quadratic_state_control_cost_function(
        options.x_star,
        options.u_star,
        options.Q * options.state_cost_scaling,
        options.R,
    )
    final_cost_function = make_quadratic_cost_function(
        options.x_star, options.Q * options.final_cost_scaling
    )
    relaxed_final_cost_function = make_quadratic_cost_function(
        options.x_star, options.Q * options.relax_final_state_constraint_multiplier
    )

    # while there are outgoing edges
    # while len(edges_out) != 0:
    while horizon_index < options.N:
        current_potential = current_vertex.cost_at_point(current_state, gcs.solution)

        # for each edge, get the next vertex, then solve an optimization to evaluate that edge
        for edge0 in edges_out:
            x0, v0 = current_state, current_vertex
            dynamics_function_0 = edge0.dynamics_function
            # next vertex
            v1 = edge0.right
            # do a two step lookahead, unless request otherwise
            edges_out_out = [gcs.edges[name] for name in v1.edges_out]
            if horizon_index < options.N - 1 and options.two_step_lookahead:
                # second step
                for edge1 in edges_out_out:
                    dynamics_function_1 = edge1.dynamics_function

                    # next-next vertex and its potential
                    v2 = edge1.right
                    # potential = gcs.solution.GetSolution(v2.potential)
                    if options.add_noise:
                        potential = v2.get_quadratic_potential(gcs.solution)
                    else:
                        potential = gcs.solution.GetSolution(v2.potential)

                    prog = MathematicalProgram()
                    # define control input
                    u0 = prog.NewContinuousVariables(current_vertex.control_dim, "u0")
                    u1 = prog.NewContinuousVariables(current_vertex.control_dim, "u1")
                    # define states
                    x1 = dynamics_function_0(x0, u0)
                    x2 = dynamics_function_1(x1, u1)

                    x2_sub_dict = {v2.x[i]: x2[i] for i in range(v2.state_dim)}

                    if not isinstance(potential, Expression):
                        potential = potential.ToExpression()
                    expr = potential.Substitute(x2_sub_dict)
                    expr += rolling_cost_function(x0, u0)
                    expr += rolling_cost_function(x1, u1)
                    prog.AddCost(expr)

                    # subject to constraints
                    try:
                        # control constraints
                        prog.AddLinearConstraint(
                            le(v0.control_set.A().dot(u0), v0.control_set.b())
                        )
                        prog.AddLinearConstraint(
                            le(v1.control_set.A().dot(u1), v1.control_set.b())
                        )
                        # state constraints
                        prog.AddLinearConstraint(
                            le(v1.state_set.A().dot(x1), v1.state_set.b())
                        )

                        if (
                            horizon_index == options.N - 2
                        ) and options.relax_final_state_constraint:
                            # prog.AddLinearConstraint( le(v1.state_set.A().dot(x2), v1.state_set.b()) )
                            prog.AddCost(relaxed_final_cost_function(x2))
                        else:
                            prog.AddLinearConstraint(
                                le(v2.state_set.A().dot(x2), v2.state_set.b())
                            )

                    except RuntimeError:
                        # failed to add constraints -- bc some constraints are infeasible.
                        results_out.append(
                            (float("inf"), None, None, None, float("inf"))
                        )
                        continue

                    # solve with snopt
                    solver = SnoptSolver()
                    solution = solver.Solve(prog)  # type: MathematicalProgramResult

                    # check if solution is success
                    if solution.is_success():
                        cost = solution.get_optimal_cost()
                        u_value = solution.GetSolution(u0)
                        y_value = dynamics_function_0(x0, u_value)
                        # note -- rolling cost is only a single step cost
                        rolling_cost = rolling_cost_function(x0, u_value)
                        min_cost_value = cost
                        # if complimentary slackness -- evaluate how far the minimum is from current potential
                        if options.complimentary_slackness_policy:
                            min_cost_value = np.abs(cost - current_potential)
                        results_out.append(
                            (
                                min_cost_value,
                                v1,
                                u_value,
                                y_value,
                                rolling_cost,
                                edge0.name,
                            )
                        )
                    else:
                        # failed to solve
                        results_out.append(
                            (float("inf"), None, None, None, float("inf"))
                        )

            else:
                # single step lookahead
                prog = MathematicalProgram()
                # define control input, dynamics
                u0 = prog.NewContinuousVariables(current_vertex.control_dim, "u0")
                x1 = dynamics_function_0(x0, u0)

                # form the cost
                # potential = gcs.solution.GetSolution(v1.potential)
                if options.add_noise:
                    potential = v1.get_quadratic_potential(gcs.solution)
                else:
                    potential = gcs.solution.GetSolution(v1.potential)

                x1_sub_dict = {v1.x[i]: x1[i] for i in range(v1.state_dim)}
                if not isinstance(potential, Expression):
                    potential = potential.ToExpression()
                expr = potential.Substitute(x1_sub_dict)
                expr += rolling_cost_function(x0, u0)
                prog.AddCost(expr)
                # subject to constraints
                try:
                    # control constraints
                    prog.AddLinearConstraint(
                        le(v0.control_set.A().dot(u0), v0.control_set.b())
                    )
                    # state constraints
                    if (
                        horizon_index == options.N - 1
                    ) and options.relax_final_state_constraint:
                        prog.AddCost(relaxed_final_cost_function(x1))
                    else:
                        prog.AddLinearConstraint(
                            le(v1.state_set.A().dot(x1), v1.state_set.b())
                        )
                except RuntimeError:
                    # failed to add constraints -- bc some constraints are infeasible.
                    results_out.append((float("inf"), None, None, None, float("inf")))
                    continue

                solver = SnoptSolver()
                solution = solver.Solve(
                    prog
                )  # type: MathematicalProgramResult # should add initial guess
                # check if solution is success
                if solution.is_success():
                    cost = solution.get_optimal_cost()
                    u_value = solution.GetSolution(u0)
                    y_value = dynamics_function_0(x0, u_value)
                    rolling_cost = rolling_cost_function(x0, u_value)
                    min_cost_value = cost
                    if options.complimentary_slackness_policy:
                        min_cost_value = np.abs(cost - current_potential)
                    results_out.append(
                        (min_cost_value, v1, u_value, y_value, rolling_cost)
                    )
                else:
                    if horizon_index >= 10:
                        WARN("solving", horizon_index, start_state)
                    results_out.append((float("inf"), None, None, None, float("inf")))

        # determine the best edge
        results_out = sorted(results_out, key=lambda x: x[0])
        result = results_out[0]
        # if all edges have infinite cost -- there is no solution
        if np.isinf(result[0]):
            ERROR("no edges at index", horizon_index, start_state)
            return np.array(x_trajectory), np.array(u_trajectory), float("inf")

        # update everything
        if options.propagate_linearized:
            current_vertex = result[1]
            current_state = result[3]
            current_mode = options.system.get_mode_for_point(current_state)
            if current_mode is None:
                return np.array(x_trajectory), np.array(u_trajectory), float("inf")
            # assert current_mode is not None, ERROR("current state is not inside any mode", current_state)
            x_trajectory.append(result[3])
            u_trajectory.append(result[2])
            mode_trajectory.append(current_mode)
            total_cost += result[4]
        else:
            # TODO: check me for bugs
            # get this edge in the graph
            mode_name = options.system.get_mode_for_point(current_state)
            assert mode_name is not None, ERROR(
                "current state inside no mode", current_state
            )
            # get nonlinear dynamics for this edge
            next_state, next_mode_name = options.system.propagate_nonlinear(
                current_state, result[2], options.dt
            )
            if next_mode_name is None:
                return np.array(x_trajectory), np.array(u_trajectory), float("inf")
            # get current vertex
            if horizon_index < options.N - 1:
                current_vertex = gcs.vertices[
                    get_vertex_name(horizon_index + 1, next_mode_name)
                ]
            current_state = next_state
            current_mode = next_mode_name
            x_trajectory.append(next_state)
            u_trajectory.append(result[2])
            mode_trajectory.append(next_mode_name)
            total_cost += result[4]

        edges_out, results_out = [
            gcs.edges[name] for name in current_vertex.edges_out
        ], []
        horizon_index += 1

    # add final state cost
    total_cost += final_cost_function(current_state)
    return np.array(x_trajectory), np.array(u_trajectory), total_cost

import typing as T
import numpy as np
import numpy.typing as npt
import logging

import plotly.graph_objects as go  # pylint: disable=import-error
from plotly.express.colors import sample_colorscale  # pylint: disable=import-error

from program_options import ProgramOptions
from make_a_finite_horizion_gcs import solve_a_finite_horizon_gcs, get_vertex_name
from make_a_finite_horizion_polynomial_gcs import make_a_finite_horizon_polynomial_dual
from control_policy import control_policy
from util import timeit, INFO, YAY, ERROR, WARN, diditwork
from controls_graph import ControlsGraph
from gcs_util import get_vertex_name, make_quadratic_cost_function

logging.getLogger("drake").setLevel(logging.WARNING)

CAM_SETTINGS_UP = dict(x=0, y=0, z=1)
CAM_SETTINGS_CENTER = dict(x=0, y=0, z=0)
CAM_SETTINGS_EYE = dict(x=0.65, y=-1.8, z=1)

# TODO: the way i handle inifiniteis is bad


def stats(Z: npt.NDArray, Z_gt: npt.NDArray, name1="1", name2="2"):
    INFO("---")
    INFO(
        "# of infinities is",
        np.sum(Z == np.inf),
        "which is ",
        np.round(np.sum(Z == np.inf) / np.sum(Z >= -1), 3) * 100,
        "perecnt",
    )
    INFO("Mean cost of " + name1, np.mean(Z[Z < np.inf]))
    INFO("Median cost of " + name1, np.median(Z[Z < np.inf]))
    INFO(
        "Mean diff btwn " + name1 + " and " + name2 + "\t",
        np.mean(Z[Z < np.inf] / Z_gt[Z < np.inf]) * 100,
        "%",
    )
    INFO(
        "Median diff btwn " + name1 + " and " + name2 + "\t",
        np.median(Z[Z < np.inf] / Z_gt[Z < np.inf]) * 100,
        "%",
    )
    INFO("# points in " + name1 + " ABOVE " + name2 + "\t", np.sum(Z > Z_gt + 1e-5))
    INFO("# points in " + name1 + " BELOW " + name2 + "\t", np.sum(Z < Z_gt - 1e-5))
    INFO("---")


def mode_sequence_from_traj(system, x_traj: npt.NDArray):
    """
    Returns a string of zeros and ones.
    0 for no-contact mode, 1 for contact mode
    """
    # TODO: fix me
    mode_sequence = []
    for x in x_traj:
        mode_name = None
        for v in graph.vertices.values():
            if v.is_point_in_set(x):
                if mode_name is not None:
                    WARN("ambiguous mode for state ", x)
                mode_name = v.name
        if mode_name is None:
            ERROR("point is in no mode", x)
        mode_sequence.append(mode_name)
    return "".join(mode_sequence)


# -------------------------------------------------------------------------------------------------
# making results data
# -------------------------------------------------------------------------------------------------


class TrajectoryResult:
    def __init__(
        self,
        x_traj: npt.NDArray,
        u_traj: npt.NDArray,
        trajectory_cost: float,
        lower_bound_on_cost: float,
        mode_sequence_index: int,
        options: ProgramOptions,
    ) -> None:
        self.x_traj = x_traj
        self.u_traj = u_traj
        self.trajectory_cost = trajectory_cost
        self.lower_bound_on_cost = lower_bound_on_cost
        self.mode_sequence_index = mode_sequence_index
        self.options = options

        self.x0 = x_traj[0, :]
        self.x_final = x_traj[-1, :]
        self.verify_that_trajectory_is_correct(options.system)

    def compute_cost(self, x_traj, u_traj):  # TODO: fix this
        rolling_state_cost = make_quadratic_cost_function(
            self.options.x_star, self.options.Q * self.options.state_cost_scaling
        )
        final_state_cost = make_quadratic_cost_function(
            self.options.x_star, self.options.Q * self.options.final_cost_scaling
        )
        rolling_control_cost = make_quadratic_cost_function(
            self.options.u_star, self.options.R
        )

        total_cost = 0
        for i in range(len(u_traj)):
            total_cost += rolling_state_cost(x_traj[i])
            total_cost += rolling_control_cost(u_traj[i])
        total_cost += final_state_cost(x_traj[-1])
        return total_cost

    def print_costs(self, system):
        x_traj_rollout = system.rollout_trajectory(
            self.x0, self.u_traj, self.options.dt, self.options.propagate_linearized
        )
        rollout_cost = self.compute_cost(x_traj_rollout, self.u_traj)
        cost_of_x_traj = self.compute_cost(self.x_traj, self.u_traj)
        INFO("Rollout cost: ", rollout_cost)
        INFO("Traj Cost: ", cost_of_x_traj)
        INFO("Optimization cost:", self.trajectory_cost)

    def verify_that_trajectory_is_correct(self, system) -> bool:
        correct = True
        x_traj_rollout = system.rollout_trajectory(
            self.x0, self.u_traj, self.options.dt, self.options.propagate_linearized
        )
        assert x_traj_rollout.shape == self.x_traj.shape
        atol = 1e-5

        # uncomment to check if trajectories are the same
        # if not np.allclose(x_traj_rollout, self.x_traj, atol=atol):
        #     WARN("Control rollout doesn't match provided trajectory")
        #     print(np.isclose(x_traj_rollout, self.x_traj, atol=atol).T)
        #     WARN("rollout:")
        #     WARN(x_traj_rollout.T)
        #     WARN("traj:")
        #     WARN(self.x_traj.T)
        #     correct = False

        rollout_cost = self.compute_cost(
            x_traj_rollout, self.u_traj
        )  # TODO: rollout probably uses pendulum; should use system; fix
        cost_of_x_traj = self.compute_cost(self.x_traj, self.u_traj)

        if not np.allclose(cost_of_x_traj, self.trajectory_cost, atol=atol):
            if self.trajectory_cost <= 1e5:
                WARN("optimization cost does not match cost computed from trajectory")
                WARN("optimization:", self.trajectory_cost)
                WARN("computed:", cost_of_x_traj)
                correct = False

        if not np.allclose(cost_of_x_traj, rollout_cost, atol=atol):
            WARN("rollout cost does not match cost computed from trajectory")
            WARN("rollout:", rollout_cost)
            WARN("computed:", cost_of_x_traj)
            correct = False

        if not correct:
            WARN("mismatch for x0 value of:", self.x0)

        return correct


def make_ground_truth_results(
    X: npt.NDArray, Y: npt.NDArray, graph, options: ProgramOptions
) -> T.List[TrajectoryResult]:
    """
    Solves GCS to attain ground truth values at individual points.

    Returns:
    - results -- a list of (state_trajectory, trajectory_cost, trajectory_cost, mode_index)
    - number of modes
    """

    # dectionary: mode_sequence to integer number
    mode_sequences = dict()
    results = []
    mode_seq_index = 0
    # generate trajectories and determine modes
    # make_one_plot = True
    for q in X:
        for dq in Y:
            # solve gcs, get a trajectory
            options.x0 = np.array([q, dq])
            x_traj, u_traj, cost = solve_a_finite_horizon_gcs(graph, options)

            # for feasible trajectories
            if len(x_traj) > 0:
                mode_seq = mode_sequence_from_traj(graph, x_traj)
                # mark the mode sequence
                if mode_seq not in mode_sequences:
                    mode_sequences[mode_seq] = mode_seq_index
                    results.append(
                        TrajectoryResult(
                            x_traj, u_traj, cost, cost, mode_seq_index, options
                        )
                    )
                    mode_seq_index += 1
                else:
                    results.append(
                        TrajectoryResult(
                            x_traj,
                            u_traj,
                            cost,
                            cost,
                            mode_sequences[mode_seq],
                            options,
                        )
                    )

    return results


def make_control_policy_results(
    X: npt.NDArray,
    Y: npt.NDArray,
    start_mode_name: str,
    graph: ControlsGraph,
    options: ProgramOptions,
) -> T.List[TrajectoryResult]:
    """
    Solves a polynomial GCS to obtain cost-to-go lower bound;
    results are generated from an induced control policy.

    Returns:
    - results -- a list of (state_trajectory, trajectory-cost, lower-bound-on-cost, mode_index)
    - number of modes
    """

    # solve a polynomial GCS
    gcs, t = make_a_finite_horizon_polynomial_dual(graph, options)
    s = gcs.vertices[get_vertex_name(0, start_mode_name)]

    solution = gcs.Solve(options)

    if not solution.is_success():
        WARN("Failed to solve the SDP")

    # evaluate potentials at the grid
    poly = solution.GetSolution(s.potential)

    # dectionary: mode_sequence to integer number
    mode_sequences = dict()
    results = []
    mode_seq_index = 0
    # generate trajectories and determine modes
    for q in X:
        for dq in Y:
            # get the lower bound by evaluating potential
            lower_bound_on_cost = poly.Evaluate({s.x[0]: q, s.x[1]: dq})

            # get the upper bound by evaluating the control policy
            x0 = np.array([q, dq])
            x_traj, u_traj, upper_bound_on_cost = control_policy(
                gcs, graph, x0, options
            )

            # for feasible trajectories
            if len(x_traj) > 0:
                mode_seq = mode_sequence_from_traj(graph, x_traj)  #
                # mark the mode sequence
                if mode_seq not in mode_sequences:
                    mode_sequences[mode_seq] = mode_seq_index
                    results.append(
                        TrajectoryResult(
                            x_traj,
                            u_traj,
                            upper_bound_on_cost,
                            lower_bound_on_cost,
                            mode_seq_index,
                            options,
                        )
                    )
                    mode_seq_index += 1
                else:
                    results.append(
                        TrajectoryResult(
                            x_traj,
                            u_traj,
                            upper_bound_on_cost,
                            lower_bound_on_cost,
                            mode_sequences[mode_seq],
                            options,
                        )
                    )

    return results


# -------------------------------------------------------------------------------------------------
# phase plots
# -------------------------------------------------------------------------------------------------


def draw_phase_plots_from_results(
    results: T.List[TrajectoryResult],
    plot_trajectories: bool = True,
    plot_a_subset: npt.NDArray = None,
    two_walls=False,
):
    fig = go.Figure()

    num_of_diff_mode_sequences = int(
        np.max(np.array([result.mode_sequence_index for result in results])) + 1
    )
    colors = sample_colorscale("Plasma", num_of_diff_mode_sequences)

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            marker_symbol="square",
            marker=dict(size=12, line=dict(width=10, color="black")),
            showlegend=False,
        )
    )
    fig.add_vline(x=0.1)
    if two_walls:
        fig.add_vline(x=-0.1)

    counter = 0
    for result in results:
        plot_this_traj = True
        if plot_a_subset is not None:
            plot_this_traj = False
            for val in plot_a_subset:
                if np.allclose(val, result.x0):
                    plot_this_traj = True
                    break

        if plot_this_traj:
            counter += 1
            if plot_trajectories:
                fig.add_trace(
                    go.Scatter(
                        x=result.x_traj[:, 0],
                        y=result.x_traj[:, 1],
                        line=dict(color=colors[result.mode_sequence_index]),
                        showlegend=False,
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=[result.x0[0]],
                    y=[result.x0[1]],
                    marker_symbol="circle",
                    marker=dict(
                        size=5,
                        line=dict(width=7, color=colors[result.mode_sequence_index]),
                    ),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[result.x_final[0]],
                    y=[result.x_final[1]],
                    marker_symbol="square",
                    marker=dict(
                        size=5,
                        line=dict(width=7, color=colors[result.mode_sequence_index]),
                    ),
                    showlegend=False,
                )
            )

    if plot_a_subset is not None:
        WARN(len(plot_a_subset), " points in the subset")
        WARN(counter, " points plotted")

    # add a dummy trace for the modes
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[0, 0],
            mode="markers",
            marker=dict(
                size=0,
                color=[0, num_of_diff_mode_sequences - 1],
                colorscale="Plasma",
                colorbar={"title": "Mode Sequence", "dtick": 1},
                showscale=True,
            ),
            showlegend=False,
            hoverinfo="none",
            visible=False,
        )
    )

    fig.update_layout(
        title="Trajectories & Mode Sequences (total of "
        + str(num_of_diff_mode_sequences)
        + " different sequences) ",
        width=800,
        height=800,
        autosize=False,
    )
    fig.update_layout(xaxis_title="q, rad", yaxis_title="dq, rad/s")
    fig.show()


# -------------------------------------------------------------------------------------------------
# meshgrids
# -------------------------------------------------------------------------------------------------


def make_meshgrid_from_results(
    X: npt.NDArray, Y: npt.NDArray, results: T.List[TrajectoryResult]
) -> T.Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    XX, YY = np.meshgrid(X, Y)
    Z_upper = np.zeros(XX.shape)
    Z_lower = np.zeros(XX.shape)
    C = np.zeros(XX.shape)
    # note the indexing shenanigans
    j, i = 0, 0
    for result in results:
        Z_upper[j, i] = result.trajectory_cost
        Z_lower[j, i] = result.lower_bound_on_cost
        C[j, i] = str(result.mode_sequence_index)
        j += 1
        if j == len(Y):
            j, i = 0, i + 1
    return XX, YY, Z_upper, Z_lower, C


def draw_meshgrid_ground_truth_with_modes(
    X: npt.NDArray, Y: npt.NDArray, results: T.List[TrajectoryResult]
):
    XX, YY, Z_gt, _, C = make_meshgrid_from_results(X, Y, results)

    fig = go.Figure(
        data=[
            go.Surface(
                x=XX,
                y=YY,
                z=Z_gt,
                surfacecolor=C,
                name="Ground Truth Cost",
                showscale=True,
                showlegend=False,
                colorbar={"title": "Mode Sequence", "dtick": 1},
            )
        ]
    )
    fig.update_layout(
        title="Ground Truth with Mode Sequences",
        width=1000,
        height=1000,
        autosize=False,
        scene=dict(
            xaxis=dict(title="q"), yaxis=dict(title="dq"), zaxis=dict(title="potential")
        ),
    )
    camera = dict(
        up=CAM_SETTINGS_UP,
        center=CAM_SETTINGS_CENTER,
        eye=CAM_SETTINGS_EYE,
    )
    fig.update_layout(scene_camera=camera)
    fig.show()


def draw_meshgrid_from_results(
    X: npt.NDArray,
    Y: npt.NDArray,
    results: T.List[TrajectoryResult],
    draw_with_modes: bool = True,
):
    XX, YY, Z_upper, Z_lower, C = make_meshgrid_from_results(X, Y, results)
    stats(Z_upper, Z_lower, name1="upper bound", name2="lower bound")

    if draw_with_modes:
        fig = go.Figure(
            data=[
                go.Surface(
                    x=XX,
                    y=YY,
                    z=Z_lower,
                    colorscale=[[0, "blue"], [1, "blue"]],
                    name="Lower Bound",
                    showscale=False,
                ),
                go.Surface(
                    x=XX,
                    y=YY,
                    z=Z_upper,
                    surfacecolor=C,
                    name="Upper Bound",
                    showscale=True,
                    showlegend=False,
                    colorbar={"title": "Mode Sequence", "dtick": 1},
                ),
            ]
        )
    else:
        fig = go.Figure(
            data=[
                go.Surface(
                    x=XX,
                    y=YY,
                    z=Z_lower,
                    colorscale=[[0, "blue"], [1, "blue"]],
                    name="Lower Bound",
                    showscale=False,
                    showlegend=True,
                    lighting={"ambient": 0.5, "diffuse": 0.5},
                ),
                go.Surface(
                    x=XX,
                    y=YY,
                    z=Z_upper,
                    colorscale=[[0, "red"], [1, "red"]],
                    name="Upper bound",
                    showscale=False,
                    showlegend=True,
                    lighting={"ambient": 0.5, "diffuse": 0.5},
                ),
            ]
        )

    fig.update_layout(
        title="Lower vs Ground-truth vs Upper",
        width=1000,
        height=1000,
        autosize=False,
        scene=dict(
            xaxis=dict(title="q, rad"),
            yaxis=dict(title="dq, rad/s"),
            zaxis=dict(title="potential or cost-to-go"),
        ),
    )
    camera = dict(
        up=CAM_SETTINGS_UP,
        center=CAM_SETTINGS_CENTER,
        eye=CAM_SETTINGS_EYE,
    )
    fig.update_layout(scene_camera=camera)
    fig.show()


def compare_meshgrids(
    X: npt.NDArray,
    Y: npt.NDArray,
    results_gt: T.List[TrajectoryResult],
    results: T.List[TrajectoryResult],
):
    _, _, Z_gt, _, _ = make_meshgrid_from_results(X, Y, results_gt)
    XX, YY, Z_upper, Z_lower, _ = make_meshgrid_from_results(X, Y, results)
    stats(Z_upper, Z_gt, name1="upper bound", name2="gt")

    fig = go.Figure(
        data=[
            go.Surface(
                x=XX,
                y=YY,
                z=Z_gt,
                colorscale=[[0, "green"], [1, "green"]],
                name="Ground Truth",
                showscale=False,
                showlegend=True,
                lighting={"ambient": 0.5, "diffuse": 0.5},
            ),
            go.Surface(
                x=XX,
                y=YY,
                z=Z_lower,
                colorscale=[[0, "blue"], [1, "blue"]],
                name="Lower Bound",
                showscale=False,
                showlegend=True,
                lighting={"ambient": 0.5, "diffuse": 0.5},
            ),
            go.Surface(
                x=XX,
                y=YY,
                z=Z_upper,
                colorscale=[[0, "red"], [1, "red"]],
                name="Upper bound",
                showscale=False,
                showlegend=True,
                lighting={"ambient": 0.5, "diffuse": 0.5},
            ),
        ]
    )
    fig.update_layout(
        title="Lower vs Ground-truth vs Upper",
        width=1000,
        height=1000,
        autosize=False,
        scene=dict(
            xaxis=dict(title="q"), yaxis=dict(title="dq"), zaxis=dict(title="potential")
        ),
    )
    camera = dict(
        up=CAM_SETTINGS_UP,
        center=CAM_SETTINGS_CENTER,
        eye=CAM_SETTINGS_EYE,
    )
    fig.update_layout(scene_camera=camera)
    fig.show()


def compute_a_databag(
    X: npt.NDArray,
    Y: npt.NDArray,
    start_mode_name: str,
    graph: ControlsGraph,
    options: ProgramOptions,
    ground_truth: bool = False,
    color: str = "green",
):
    # make the name
    if options.push_up_box_at_uniform_sample_of_points:
        push = "sample"
    else:
        push = "integral"

    name = (
        "PD: "
        + str(options.potential_poly_deg)
        + ", MD: "
        + str(options.max_constraint_degree)
        + ", "
        + push
    )
    if ground_truth:
        name = "Ground Truth"

    # copmute things
    INFO("Computing: ", name)
    if ground_truth:
        results = make_ground_truth_results(X, Y, graph, options)
    else:
        results = make_control_policy_results(X, Y, start_mode_name, graph, options)
    XX, YY, Z_upper, Z_lower, C = make_meshgrid_from_results(X, Y, results)

    # return a databag
    if ground_truth:
        return Databag(
            results, XX, YY, None, None, Z_upper, C, name, ground_truth, "green"
        )
    else:
        return Databag(
            results, XX, YY, Z_upper, Z_lower, None, C, name, ground_truth, color
        )


class Databag:
    def __init__(
        self,
        results: T.List[TrajectoryResult],
        XX: npt.NDArray,
        YY: npt.NDArray,
        Z_uppper: npt.NDArray,
        Z_lower: npt.NDArray,
        Z_gt: npt.NDArray,
        C: npt.NDArray,
        name: str,
        ground_truth: bool,
        color: str,
    ):
        self.results = results
        self.XX = XX
        self.YY = YY
        self.Z_upper = Z_uppper
        self.Z_lower = Z_lower
        self.Z_gt = Z_gt
        self.C = C
        self.name = name
        self.color = color
        self.ground_truth = ground_truth

    def set_color(self, color):
        self.color = color


def do_magic_on_databags(
    databags: T.List[Databag],
    gt_databag: Databag = None,
    compare_upper_to_lower: bool = False,
    compare_upper_to_gt: bool = False,
    compare_gt_to_lower: bool = False,
    plot_lower: bool = False,
    plot_upper: bool = False,
    plot_gt: bool = False,
    plot_modes: bool = False,
    colors: T.List[str] = None,
):

    # if passed a single databag, put it in a list
    justone = False
    if isinstance(databags, Databag):
        databags = [databags]
        justone = True

    if compare_upper_to_lower:
        for databag in databags:
            INFO(databag.name, ":")
            stats(
                databag.Z_upper,
                databag.Z_lower,
                name1="upper bound",
                name2="lower bound",
            )

    WARN("--------")

    if compare_upper_to_gt:
        assert gt_databag is not None
        for databag in databags:
            INFO(databag.name, ":")
            stats(databag.Z_upper, gt_databag.Z_gt, name1="upper bound", name2="gt")

    WARN("--------")

    if compare_gt_to_lower:
        assert gt_databag is not None
        for databag in databags:
            INFO(databag.name, ":")
            stats(gt_databag.Z_gt, databag.Z_lower, name1="gt", name2="Z_lower")

    WARN("--------")

    fig = go.Figure()
    if plot_gt:
        if plot_modes:
            INFO("Modes in ", gt_databag.name, "is \t", int(np.max(gt_databag.C) + 1))
            fig.add_trace(
                go.Surface(
                    x=gt_databag.XX,
                    y=gt_databag.YY,
                    z=gt_databag.Z_gt,
                    surfacecolor=gt_databag.C,
                    name=gt_databag.name,
                    showscale=False,
                    showlegend=True,
                    colorbar={"title": "Mode Sequence", "dtick": 1},
                )
            )
        else:
            fig.add_trace(
                go.Surface(
                    x=gt_databag.XX,
                    y=gt_databag.YY,
                    z=gt_databag.Z_gt,
                    colorscale=[[0, "green"], [1, "green"]],
                    name=gt_databag.name,
                    showscale=False,
                    showlegend=True,
                    lighting={"ambient": 0.5, "diffuse": 0.5},
                )
            )

    if plot_lower:
        index = 0
        for databag in databags:
            index += 1
            if plot_modes:
                INFO("Modes in ", databag.name, "is \t", int(np.max(databag.C) + 1))
                fig.add_trace(
                    go.Surface(
                        x=databag.XX,
                        y=databag.YY,
                        z=databag.Z_lower,
                        surfacecolor=databag.C,
                        name="Lower " + databag.name,
                        showscale=False,
                        showlegend=True,
                        colorbar={"title": "Mode Sequence", "dtick": 1},
                    )
                )
            else:
                color = databag.color
                if justone:
                    color = "blue"
                if colors is not None:
                    color = colors[index - 1]
                fig.add_trace(
                    go.Surface(
                        x=databag.XX,
                        y=databag.YY,
                        z=databag.Z_lower,
                        colorscale=[[0, color], [1, color]],
                        name="Lower " + databag.name,
                        showscale=False,
                        showlegend=True,
                        lighting={"ambient": 0.5, "diffuse": 0.5},
                    )
                )

    if plot_upper:
        index = 0
        for databag in databags:
            index += 1
            if plot_modes:
                INFO("Modes in ", databag.name, "is \t", int(np.max(databag.C) + 1))
                fig.add_trace(
                    go.Surface(
                        x=databag.XX,
                        y=databag.YY,
                        z=databag.Z_upper,
                        surfacecolor=databag.C,
                        name="Upper " + databag.name,
                        showscale=False,
                        showlegend=True,
                        colorbar={"title": "Mode Sequence", "dtick": 1},
                    )
                )
            else:
                color = databag.color
                if justone:
                    color = "red"
                if colors is not None:
                    color = colors[index - 1]
                fig.add_trace(
                    go.Surface(
                        x=databag.XX,
                        y=databag.YY,
                        z=databag.Z_upper,
                        colorscale=[[0, color], [1, color]],
                        name="Upper " + databag.name,
                        showscale=False,
                        showlegend=True,
                        lighting={"ambient": 0.5, "diffuse": 0.5},
                    )
                )

    fig.update_layout(
        width=1300,
        height=1000,
        autosize=False,
        scene=dict(
            xaxis=dict(title="q"),
            yaxis=dict(title="dq"),
            zaxis=dict(title="potential or cost-to-go"),
        ),
    )
    camera = dict(
        up=CAM_SETTINGS_UP,
        center=CAM_SETTINGS_CENTER,
        eye=CAM_SETTINGS_EYE,
    )
    fig.update_layout(scene_camera=camera)
    if plot_lower or plot_upper or plot_gt:
        fig.show()


# -------------------------------------------------------------------------------------------------
# old
# -------------------------------------------------------------------------------------------------


def get_grid_and_ground_truth_values(
    X: npt.NDArray, Y: npt.NDArray, graph: ControlsGraph, options: ProgramOptions
) -> T.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    # define the grid of points
    XX, YY = np.meshgrid(X, Y)

    gt_options = options
    gt_options.use_convex_relaxation = True
    gt_options.max_rounded_paths = 25

    # compute ground truth value at a point
    def get_ground_truth_values(q, dq):
        gt_options.x0 = np.array([q, dq])
        return solve_a_finite_horizon_gcs(graph, gt_options, just_cost=True)

    # vectorize the function
    get_ground_truth_grid = np.vectorize(get_ground_truth_values)
    # compute ground truth values over entire grid
    Z_ground_truth = get_ground_truth_grid(XX, YY)

    return XX, YY, Z_ground_truth


def solve_polynomial_gcs_and_get_a_grid_of_values(
    XX: npt.NDArray,
    YY: npt.NDArray,
    start_mode_name: str,
    graph: ControlsGraph,
    options: ProgramOptions,
    get_upper_bounds_too: bool,
) -> T.Tuple[npt.NDArray, npt.NDArray]:

    # solve the polynomial GCS program
    gcs, t = make_a_finite_horizon_polynomial_dual(graph, options)
    s = gcs.vertices[get_vertex_name(0, start_mode_name)]

    solution = gcs.Solve(options)
    if not solution.is_success():
        ERROR("Failed to solve the SDP")
        # return None, None
    # evaluate potentials at the grid
    poly = solution.GetSolution(s.potential)
    evaluate_potentials = np.vectorize(
        lambda x, y: poly.Evaluate({s.x[0]: x, s.x[1]: y})
    )
    Z_lower_bound = evaluate_potentials(XX, YY)

    # obtain upper bounds by evaluating the control policy
    Z_upper_bound = None
    if get_upper_bounds_too:

        def get_upper_bound(q: float, dq: float) -> float:
            x0 = np.array([q, dq])
            _, _, upper_bound_on_cost = control_policy(gcs, s, x0, options)
            return upper_bound_on_cost

        get_upper_bound_vectorized = np.vectorize(get_upper_bound)
        Z_upper_bound = get_upper_bound_vectorized(XX, YY)

    return Z_lower_bound, Z_upper_bound


def plot_lower_upper_ground_truth(
    XX: npt.NDArray,
    YY: npt.NDArray,
    Z_gt: npt.NDArray,
    Z_lower: npt.NDArray,
    Z_upper: npt.NDArray,
    degree: int,
):
    stats(Z_upper, Z_gt, name1="upper bound", name2="gt")
    fig = go.Figure(
        data=[
            go.Surface(
                x=XX,
                y=YY,
                z=Z_gt,
                colorscale=[[0, "green"], [1, "green"]],
                name="Ground Truth",
                showscale=False,
                showlegend=True,
                lighting={"ambient": 0.5, "diffuse": 0.5},
            ),
            go.Surface(
                x=XX,
                y=YY,
                z=Z_lower,
                colorscale=[[0, "blue"], [1, "blue"]],
                name="Lower Bound, Degree" + str(degree),
                showscale=False,
                showlegend=True,
                lighting={"ambient": 0.5, "diffuse": 0.5},
            ),
            go.Surface(
                x=XX,
                y=YY,
                z=Z_upper,
                colorscale=[[0, "red"], [1, "red"]],
                name="Upper Bound, Degree" + str(degree),
                showscale=False,
                showlegend=True,
                lighting={"ambient": 0.5, "diffuse": 0.5},
            ),
        ]
    )
    fig.update_layout(
        title="Lower vs Ground-truth vs Upper, Degree " + str(degree),
        width=1000,
        height=1000,
        autosize=False,
        scene=dict(
            xaxis=dict(title="q"), yaxis=dict(title="dq"), zaxis=dict(title="potential")
        ),
    )
    camera = dict(
        up=CAM_SETTINGS_UP,
        center=CAM_SETTINGS_CENTER,
        eye=CAM_SETTINGS_EYE,
    )
    fig.update_layout(scene_camera=camera)
    fig.show()


def evaluate_polynomial_compare_to_ground_truth(
    X: npt.NDArray,
    Y: npt.NDArray,
    Z_gt: npt.NDArray,
    start_mode_name: str,
    graph: ControlsGraph,
    options: ProgramOptions,
):
    Z_lower, Z_upper = solve_polynomial_gcs_and_get_a_grid_of_values(
        X, Y, start_mode_name, graph, options, get_upper_bounds_too=True
    )
    stats(Z_upper, Z_gt, name1="upper bound", name2="gt")
    plot_lower_upper_ground_truth(
        X, Y, Z_gt, Z_lower, Z_upper, options.potential_poly_deg
    )

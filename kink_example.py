import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    MosekSolver,
    MosekSolverDetails,
    SolverOptions,
    CommonSolverOption,
)
from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    Point,
    ConvexSet,
    Hyperrectangle,
)
import numbers

from pydrake.symbolic import (  # pylint: disable=import-error, no-name-in-module, unused-import
    Polynomial,
    Variable,
    Variables,
    Expression,
)

import pydot

import plotly.graph_objects as go  # pylint: disable=import-error
from plotly.express.colors import sample_colorscale  # pylint: disable=import-error
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from program_options import ProgramOptions, FREE_POLY, PSD_POLY

from util import (
    timeit,
    diditwork,
    INFO,
    YAY,
    WARN,
    ERROR,
)  # pylint: disable=import-error, no-name-in-module, unused-import
from gcs_util import get_edge_name, make_quadratic_cost_function_matrices

import logging
logging.getLogger("drake").setLevel(logging.WARNING)
np.set_printoptions(suppress=True) 


from polynomial_gcs_dual import FREE_POLY, PSD_POLY
from polynomial_gcs_dual import PolynomialDualGCS, Vertex, Edge

QUADRATIC_COST = lambda x,y: (x[0]-y[0])**2


def random_uniform_graph_generator(
    num_layers: int = 5,
    x_min: float = 0,
    x_max: float = 10,
    min_blank: float = 0.5,
    max_blank: float = 1.5,
    min_region: float = 0.5,
    max_region: float = 1.5,
    min_goal_blank: float = 1,
    max_goal_blank: float = 2,
    goal_num: int = 5,
    goal_uniform: bool = False,
    options: ProgramOptions = ProgramOptions(),
    random_seed = 1,
) -> T.Tuple[PolynomialDualGCS, T.List[T.List[Vertex]]]:
    gcs = PolynomialDualGCS(options)
    # full connectivity between edges (very unnecessary)
    # TODO: random connectivity? 3 neighbour connectivity?
    np.random.seed(random_seed)

    def box(a, b):
        return Hyperrectangle([a], [b])
    
    ###############################################################
    # make vertices

    layers = []
    # add first layer
    start_vertex = gcs.AddVertex("0-0", box(x_min, x_max))
    layers.append([start_vertex])

    # for every layer
    for n in range(1, num_layers - 1):
        layer = []
        x_now = 0.0
        k = n % 2
        index = 0
        while x_now < x_max:
            # make a skip
            if k % 2 == 0:
                x_now += np.random.uniform(min_blank, max_blank, 1)[0]
            else:
                width = np.random.uniform(min_region, max_region, 1)[0]
                v_name = str(n) + "-" + str(index)
                v = gcs.AddVertex(v_name, box(x_now, min(x_now + width, x_max)))
                layer.append(v)
                index += 1
                x_now += width
            k += 1
        layers.append(layer)
            
    
    # add target potential
    zero_potential = lambda _: Expression(0)
    layer = []
    index = 0
    if goal_uniform:
        points = np.array(list(range(goal_num))+0.5) * (x_max-x_min) / goal_num
        for p in points:
            v_name = str(num_layers-1) + "-" + str(index)
            v = gcs.AddTargetVertex(v_name, Point([p]), zero_potential)
            layer.append(v)
            index += 1
    else:
        x_now = np.random.uniform(min_goal_blank, max_goal_blank, 1)[0]
        while x_now < x_max:
            v_name = str(num_layers-1) + "-" + str(index)
            v = gcs.AddTargetVertex(v_name, Point([x_now]), zero_potential)
            x_now += np.random.uniform(min_goal_blank, max_goal_blank, 1)[0]
            layer.append(v)
            index += 1
    layers.append(layer)


    ###############################################################
    # make edges
    quadratic_cost = QUADRATIC_COST
    for i, layer in enumerate(layers[:-1]):
        next_layer = layers[i+1]
        for left_v in layer:
            for right_v in next_layer:
                gcs.AddEdge(left_v, right_v, quadratic_cost)

    
    # push up on start
    gcs.MaxCostOverABox(start_vertex, [x_min], [x_max])

    # synthesize policy
    gcs.solve_policy()
    return gcs, layers

def plot_a_layered_graph(layers:T.List[T.List[Vertex]]):
    fig = go.Figure()
    def add_trace(x_min, x_max, y):
        xs = [x_min,x_max]
        ys = [y,y]
        fig.add_trace(go.Scatter(x=xs, y=ys, line=dict(color="black") ))

    y = len(layers)
    for n, layer in enumerate(layers):
        for v in layer:
            if v.set_type == Hyperrectangle:
                add_trace(v.convex_set.lb()[0], v.convex_set.ub()[0], y)
            elif v.set_type == Point:
                add_trace(v.convex_set.x()[0], v.convex_set.x()[0], y)
        y -= 1

    fig.update_layout(height=800, width=800, title_text="Graph view")
    fig.update_layout(showlegend=False)
    fig.update_layout(
        yaxis=dict(scaleanchor="x"),  # set y-axis to have the same scaling as x-axis
        yaxis2=dict(scaleanchor="x", overlaying="y", side="right"),  # set y-axis2 to have the same scaling as x-axis
    )
    return fig

def build_m_step_horizon_from_layers(gcs:PolynomialDualGCS, layers:T.List[T.List[Vertex]], m:int, start_vertex:Vertex, layer_index:int, use_0_potentials:bool = False):
    new_gcs = PolynomialDualGCS(gcs.options)
    init_vertex = new_gcs.AddVertex(start_vertex.name, start_vertex.convex_set)
    new_layers = []
    new_layers.append([init_vertex])

    # for every layer
    last_index = min(len(layers)-1, layer_index+m)
    # WARN(len(layers), last_index)
    for n in range(layer_index+1, last_index):
        layer = []
        for v in layers[n]:
            new_v = new_gcs.AddVertex(v.name, v.convex_set)
            layer.append(new_v)
        new_layers.append(layer)
            
    # add target potential
    layer = []
    for v in layers[last_index]:
        f_potential = lambda x: Expression(0)
        if not use_0_potentials:
            potential = gcs.value_function_solution.GetSolution(v.potential).ToExpression()
            f_potential = lambda x: potential.Substitute({v.x[i]: x[i] for i in range(v.state_dim)})
        new_v = new_gcs.AddTargetVertex(v.name, v.convex_set, f_potential)
        layer.append(new_v)
    new_layers.append(layer)

    # make edges
    quadratic_cost = QUADRATIC_COST
    for i, layer in enumerate(new_layers[:-1]):
        next_layer = new_layers[i+1]
        for left_v in layer:
            for right_v in next_layer:
                new_gcs.AddEdge(left_v, right_v, quadratic_cost)

    return new_gcs

def plot_m_step_horizon_from_layers(gcs:PolynomialDualGCS, layers:T.List[T.List[Vertex]], m:int, start_vertex:Vertex, layer_index:int, dx:float=0.1):
    new_gcs = build_m_step_horizon_from_layers(gcs, layers, m, start_vertex, layer_index, use_0_potentials=False)
    x, y, _= new_gcs.get_true_cost_for_region_plot_2d(start_vertex.name, dx=dx)
    return x, y

def rollout_m_step_policy(gcs:PolynomialDualGCS, layers:T.List[T.List[Vertex]], m:int, vertex:Vertex, point:npt.NDArray, layer_index:int,use_0_potentials:bool=False) -> float:
    if layer_index < len(layers)-1:
        next_vertex, next_point = get_next_action(gcs, layers, m, vertex, point, layer_index, use_0_potentials=use_0_potentials)
        cost = rollout_m_step_policy(gcs, layers, m, next_vertex, next_point, layer_index+1, use_0_potentials=use_0_potentials)
        return QUADRATIC_COST(point, next_point) + cost
    else:
        return 0.0

def get_next_action(gcs:PolynomialDualGCS, layers:T.List[T.List[Vertex]], m:int, vertex:Vertex, point:npt.NDArray, layer_index:int, use_0_potentials: bool=False):
    # return next vertex and next point
    new_gcs = build_m_step_horizon_from_layers(gcs, layers, m, vertex, layer_index,use_0_potentials=use_0_potentials)
    _, vertex_name_path, value_path = new_gcs.solve_for_true_shortest_path(vertex.name, point)
    return gcs.vertices[vertex_name_path[1]], value_path[1]


def plot_policy_rollout(gcs:PolynomialDualGCS, layers:T.List[T.List[Vertex]], m:int, vertex:Vertex, layer_index:int, dx:float=0.1,use_0_potentials:bool=False):
    assert vertex.set_type == Hyperrectangle, "vertex not a Hyperrectangle, can't make a plot"
    assert len(vertex.convex_set.lb()) == 1, "only 1d cases for now"
    lb = vertex.convex_set.lb()[0]
    ub = vertex.convex_set.ub()[0]
    x = np.linspace(lb, ub, int((ub-lb)/dx), endpoint=True)
    y = []
    for x_val in x:
        cost = rollout_m_step_policy(gcs, layers, m, vertex, np.array([x_val]), layer_index, use_0_potentials=use_0_potentials)
        y.append(cost)
    return x, np.array(y)
    


def display_gcs_graph(gcs, graph_name="temp") -> None:
    """Visually inspect the graph. If solution acquired -- also displays the solution."""
    # if self.solution is None or not self.solution.is_success():
    graphviz = gcs.GetGraphvizString()
    # else:
        # graphviz = self.gcs.GetGraphvizString(self.solution, True, precision=2)
    data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
    data.write_png(graph_name + ".png")
    data.write_svg(graph_name + ".svg")

    # plt = Image(data.create_png())
    # display(plt)


if __name__ == "__main__":
    gcs, layers = random_uniform_graph_generator()
    v_0 = layers[0][0]
    plot_policy_rollout(gcs, layers, 1, v_0, 0)
    # x,y,_ = build_m_step_horizon_from_layers(gcs, layers, 1, v_0, 0)
    




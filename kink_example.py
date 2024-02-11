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
) -> T.Tuple[PolynomialDualGCS, T.List[T.List[Vertex]]]:
    gcs = PolynomialDualGCS(options)
    # full connectivity between edges (very unnecessary)
    # TODO: random connectivity? 3 neighbour connectivity?

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
    quadratic_cost = lambda x,y: (x[0]-y[0])**2 + 1
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


def build_m_step_horizon_from_layers(gcs:PolynomialDualGCS, layers:T.List[T.List[Vertex]], m:int, start_vertex:Vertex, layer_index:int, dx:float=0.1):
    new_gcs = PolynomialDualGCS(gcs.options)
    init_vertex = new_gcs.AddVertex(start_vertex.name, start_vertex.convex_set)
    new_layers = []
    new_layers.append([init_vertex])

    # for every layer
    last_index = min(len(layers), layer_index+m)
    for n in range(layer_index+1, last_index):
        layer = []
        for v in layers[n]:
            new_v = new_gcs.AddVertex(v.name, v.convex_set)
            layer.append(new_v)
        new_layers.append(layer)
            
    
    # add target potential
    layer = []
    for v in layers[last_index]:
        potential = gcs.value_function_solution.GetSolution(v.potential).ToExpression()
        f_potential = lambda x: potential.Substitute({v.x[i]: x[i] for i in range(v.state_dim)})
        new_v = new_gcs.AddTargetVertex(v.name, v.convex_set, f_potential)
        layer.append(new_v)
    new_layers.append(layer)

    # make edges
    quadratic_cost = lambda x,y: (x[0]-y[0])**2 + 1
    for i, layer in enumerate(new_layers[:-1]):
        next_layer = new_layers[i+1]
        for left_v in layer:
            for right_v in next_layer:
                new_gcs.AddEdge(left_v, right_v, quadratic_cost)

    x, y, ms = new_gcs.get_true_cost_for_region_plot_2d(start_vertex.name, dx=dx)
    # display_gcs_graph(new_gcs.gcs)
    return x, y, ms

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
    x,y,_ = build_m_step_horizon_from_layers(gcs, layers, 1, v_0, 0)
    




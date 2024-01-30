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
)

from pydrake.symbolic import ( # pylint: disable=import-error, no-name-in-module, unused-import
    Polynomial,
    Variable,
    Variables,
    Expression,
)  
from program_options import ProgramOptions, FREE_POLY, PSD_POLY, NOISY_DEG_2

from util import (
    timeit,
    diditwork,
    INFO,
    YAY,
    WARN,
    ERROR,
)  # pylint: disable=import-error, no-name-in-module, unused-import
from gcs_util import get_edge_name, make_quadratic_cost_function_matrices


class Vertex:
    def __init__(
        self,
        name: str,
        prog: MathematicalProgram,
        convex_set: ConvexSet,
        options: ProgramOptions,
        specific_potential: T.Callable = None, # TODO: prolly not good definition
    ):
        self.name = name
        self.potential_poly_deg = options.potential_poly_deg
        self.pot_type = options.pot_type

        # TODO: careful treatment of various sets
        # Ax <= b
        self.convex_set = convex_set # TODO: handle point exactly 
        if isinstance(convex_set, HPolyhedron):
            self.set_type = HPolyhedron
            self.state_dim = convex_set.A().shape[1]
        elif isinstance(convex_set, Point):
            self.set_type = Point
            self.state_dim = convex_set.x().shape[0]
            self.potential_poly_deg = 0 # for a constant polynomial
        else:
            self.set_type = None
            self.state_dim = None
            raise Exception("bad state set")

        self.define_variables(prog)
        self.define_set_inequalities()
        self.define_potential(prog, specific_potential)

        self.edges_in = []
        self.edges_out = []

    def add_edge_in(self, name:str):
        assert name not in self.edges_in
        self.edges_in.append(name)

    def add_edge_out(self, name:str):
        assert name not in self.edges_out
        self.edges_out.append(name)

    def define_variables(self, prog: MathematicalProgram):
        self.x = prog.NewIndeterminates(self.state_dim, "x_" + self.name)

    def define_set_inequalities(self):
        if self.set_type == HPolyhedron:
            # inequalities of the form b[i] - a.T x = g_i(x) >= 0
            A, b = self.convex_set.A(), self.convex_set.b()
            self.set_inequalities = [b[i] - A[i].dot(self.x) for i in range(len(b))]
        elif self.set_type == Point:
            self.set_inequalities = []

    def define_potential(
        self,
        prog: MathematicalProgram,
        specific_potential: T.Callable = None,
    ):
        # specific potential here is a function
        if specific_potential is not None:
            self.potential = Polynomial(specific_potential(self.x))
        else:
            # potential is a free polynomial
            if self.pot_type == FREE_POLY:
                self.potential = prog.NewFreePolynomial( self.x, self.potential_poly_deg )
            elif self.pot_type == PSD_POLY:
                assert (
                    self.potential_poly_deg % 2 == 0
                ), "can't make a PSD potential of uneven degree"
                # potential is PSD polynomial
                self.potential, _ = prog.NewSosPolynomial( self.x, self.potential_poly_deg )
            else:
                raise NotImplementedError("potential type not supported")

    @staticmethod
    def get_product_constraints(constraints):
        product_constraints = []
        for i, con_i in enumerate(constraints):
            for j in range(i + 1, len(constraints)):
                product_constraints.append(con_i * constraints[j])
        return product_constraints

    def evaluate_partial_potential_at_point(self, x: npt.NDArray):
        # needed when polynomial parameters are still optimizaiton variables
        assert len(x) == self.state_dim
        assert self.convex_set.PointInSet(x, 1e-5) # evaluate only on set
        return self.potential.EvaluatePartial(
            {self.x[i]: x[i] for i in range(self.state_dim)}
        )

    def cost_at_point(self, x: npt.NDArray, solution: MathematicalProgramResult = None):
        assert len(x) == self.state_dim
        assert self.convex_set.PointInSet(x, 1e-5)
        if solution is None:
            return self.evaluate_partial_potential_at_point(x).ToExpression()
        else:
            potential = solution.GetSolution(self.potential)
            return potential.Evaluate({self.x[i]: x[i] for i in range(self.state_dim)})

    def cost_of_uniform_integral_over_box(
        self, lb, ub, solution: MathematicalProgramResult = None
    ):
        assert len(lb) == len(ub) == self.state_dim

        # compute by integrating each monomial term
        monomial_to_coef_map = self.potential.monomial_to_coefficient_map()
        expectation = Expression(0)
        for monomial in monomial_to_coef_map.keys():
            coef = monomial_to_coef_map[monomial]
            poly = Polynomial(monomial)
            for i in range(self.state_dim):
                x_min, x_max, x_val = lb[i], ub[i], self.x[i]
                integral_of_poly = poly.Integrate(x_val)
                poly = integral_of_poly.EvaluatePartial(
                    {x_val: x_max}
                ) - integral_of_poly.EvaluatePartial({x_val: x_min})
            expectation += coef * poly.ToExpression()

        if solution is None:
            return expectation
        else:
            return solution.GetSolution(expectation)

    def cost_of_small_uniform_box_around_point(
        self, point: npt.NDArray, solution: MathematicalProgramResult = None
    ):
        assert len(point) == self.state_dim
        eps = 0.001
        return self.cost_of_uniform_integral_over_box(
            point - eps, point + eps, solution
        )


class Edge:
    def __init__(
        self,
        name: str,
        v_left: Vertex,
        v_right: Vertex,
        prog: MathematicalProgram,
        cost_function: T.Callable, # callable as a fuction of left/right variables, returns polynomial
        options: ProgramOptions,
    ):
        self.name = name
        self.left = v_left
        self.right = v_right
        self.max_constraint_degree = options.max_constraint_degree
        assert options.max_constraint_degree % 2 == 0, "SOS constraint must be PSD"
        assert options.max_constraint_degree >= self.left.potential_poly_deg
        assert options.max_constraint_degree >= self.right.potential_poly_deg
        assert options.max_constraint_degree >= 2
        self.putinar = options.putinar

        self.cost_function = cost_function 

        self.define_sos_constaint(prog)

    def define_sos_constaint(self, prog: MathematicalProgram):
        # -------------------------------------------------
        # get a bunch of variables
        x, y = self.left.x, self.right.x
        if self.left.set_type == Point:
            x = self.left.convex_set.x()
        if self.right.set_type == Point:
            y = self.right.convex_set.x()

        # -------------------------------------------------
        # define cost
        edge_cost = self.cost_function(x, y)

        # -------------------------------------------------
        # set membership through the S procedure
        # x \in X
        # y \in Y
        constraints = ( self.left.set_inequalities + self.right.set_inequalities )
        # if putinar -- add more constraints by multiplying individual constraints with each other
        if self.putinar:
            constraints += Vertex.get_product_constraints( constraints )

        # define multipliers
        multiplier_degree = self.max_constraint_degree - 2
        multipliers = [
            prog.NewSosPolynomial(vars, multiplier_degree)[0].ToExpression()
            for _ in range(len(constraints))
        ]

        s_procedure = np.array(constraints).dot(np.array(multipliers))

        # -------------------------------------------------
        # obtain right and left potentials
        right_potential = self.right.potential.ToExpression()
        left_potential = self.left.potential.ToExpression()

        # -------------------------------------------------
        # form the entire expression
        expr = edge_cost + right_potential - left_potential - s_procedure
        prog.AddSosConstraint(expr)




class PolynomialDualGCS:
    def __init__(self) -> None:
        self.vertices = dict() # type: T.Dict[str, Vertex]
        self.gcs_vertices = dict() # type: T.Dict[str, GraphOfConvexSets.Vertex]
        self.edges = dict() # type: T.Dict[str, Edge]
        self.gcs_edges = dict() # type: T.Dict[str, GraphOfConvexSets.Edge]
        self.prog = MathematicalProgram() # type: MathematicalProgram
        self.gcs = GraphOfConvexSets() # type: GraphOfConvexSets

        self.gcs_vertices["target"] = self.gcs.AddVertex(Point([0]), "target")

    def AddVertex(
        self,
        name: str,
        convex_set: HPolyhedron,
        options: ProgramOptions,
    ):
        assert name not in self.vertices
        v = Vertex(name, self.prog, convex_set, options)
        self.vertices[name] = v
        self.gcs_vertices[name] = self.gcs.AddVertex(convex_set, name)
        return v

    def MaxCostOverABox(self, vertex: Vertex, lb: npt.NDArray, ub: npt.NDArray):
        cost = -vertex.cost_of_uniform_integral_over_box(lb, ub)
        self.prog.AddLinearCost(cost)

    def MaxCostAtAPoint(self, vertex: Vertex, point):
        cost = -vertex.cost_at_point(point)
        self.prog.AddLinearCost(cost)

    def MaxCostAtSmallIntegralAroundPoint(self, vertex: Vertex, point):
        cost = -vertex.cost_at_point( vertex.cost_of_small_uniform_box_around_point(point) )
        self.prog.AddLinearCost(cost)

    def AddTargetVertex(
        self,
        name: str,
        convex_set: HPolyhedron,
        options: ProgramOptions,
        specific_potential: T.Callable,
    ):
        assert name not in self.vertices
        v = Vertex(
            name,
            self.prog,
            convex_set,
            options,
            specific_potential=specific_potential,
        )
        # building proper GCS
        self.vertices[name] = v
        self.gcs_vertices[name] = self.gcs.AddVertex(convex_set, name)
        self.gcs.AddEdge(self.gcs_vertices[name], self.gcs_vertices["target"])
        return v

    def AddEdge(
        self,
        v_left: Vertex,
        v_right: Vertex,
        cost_function: T.Callable,
        options: ProgramOptions,
    ):
        edge_name = get_edge_name(v_left.name, v_right.name)
        e = Edge(
            edge_name,
            v_left,
            v_right,
            self.prog,
            cost_function,
            options,
        )
        self.edges[edge_name] = e
        v_left.add_edge_out(edge_name)
        v_right.add_edge_in(edge_name)

        # building proper GCS
        gcs_edge = self.gcs.AddEdge(self.gcs_vertices[v_left.name], self.gcs_vertices[v_right.name], edge_name)
        self.gcs_edges[edge_name] = gcs_edge
        gcs_edge.AddCost(cost_function(gcs_edge.xu(), gcs_edge.xv()))
        return e

    def solve_policy(self, options: ProgramOptions = None) -> MathematicalProgramResult:
        if options is None:
            options = ProgramOptions()
        timer = timeit()
        mosek_solver = MosekSolver()
        solver_options = SolverOptions()

        # debug solver options
        if options.solver_debug_mode:
            solver_options.SetOption(CommonSolverOption.kPrintFileName, "solver_debug")

        # set the solver tolerance gap
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
            options.mosek_tolerance_gap,
        )
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
            options.mosek_primal_feas_gap,
        )
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
            options.mosek_dual_feas_gap,
        )
        self.solution = mosek_solver.Solve(self.prog, solver_options=solver_options)
        timer.dt("Solve")
        diditwork(self.solution)

        # debug solver options
        if options.solver_debug_mode:
            with open("solver_debug") as f:
                print(f.read())

        return self.solution


    def solve_for_true_shortest_path(self, vertex_name:str, point:npt.NDArray, options: ProgramOptions) -> T.Tuple[float, T.List[str]]:
        assert vertex_name in self.vertices
        assert self.vertices[vertex_name].convex_set.PointInSet(point, 1e-5) # evaluate only on set
        start_vertex = self.gcs.AddVertex(Point(point), "start")
        target_vertex = self.gcs_vertices["target"]

        # add edges from start point to neighbours
        for edge_name in self.vertices[vertex_name].edges_out:
            cost_function = self.edges[edge_name].cost_function
            right_vertex = self.gcs_vertices[self.edges[edge_name].right.name]
            gcs_edge = self.gcs.AddEdge(start_vertex, right_vertex)
            gcs_edge.AddCost(cost_function(gcs_edge.xu(), gcs_edge.xv()))


        gcs_options = GraphOfConvexSetsOptions()
        gcs_options.convex_relaxation = options.use_convex_relaxation
        gcs_options.max_rounding_trials = options.max_rounding_trials
        gcs_options.preprocessing = options.preprocessing

        # solve
        result = self.gcs.SolveShortestPath(start_vertex, target_vertex, gcs_options) # type: MathematicalProgramResult
        assert result.is_success()
        cost = result.get_optimal_cost()

        edge_path = self.gcs.GetSolutionPath(start_vertex, target_vertex, result) 
        vertex_name_path = [vertex_name]
        for e in edge_path:
            vertex_name_path.append(e.v().name())

        self.gcs.RemoveVertex(Point(point), "start")

        return cost, vertex_name_path


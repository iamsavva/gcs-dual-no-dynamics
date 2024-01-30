import typing as T
import numpy.typing as npt

from colorama import Fore
import time
import numpy as np
from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    SolverOptions,
    CommonSolverOption,
    IpoptSolver,
    SnoptSolver,
    MosekSolver,
    MosekSolverDetails,
    GurobiSolver,
)

from pydrake.geometry.optimization import (
    HPolyhedron,
)  # pylint: disable=import-error, no-name-in-module
from pydrake.symbolic import (
    Polynomial,
    Variable,
    Variables,
    Expression,
)  # pylint: disable=import-error, no-name-in-module, unused-import
from IPython.display import Markdown, display


def ERROR(*texts, verbose: bool = True):
    if verbose:
        print(Fore.RED + " ".join([str(text) for text in texts]))


def WARN(*texts, verbose: bool = True):
    if verbose:
        print(Fore.YELLOW + " ".join([str(text) for text in texts]))


def INFO(*texts, verbose: bool = True):
    if verbose:
        print(Fore.BLUE + " ".join([str(text) for text in texts]))


def YAY(*texts, verbose: bool = True):
    if verbose:
        print(Fore.GREEN + " ".join([str(text) for text in texts]))


def latex(prog: MathematicalProgram):
    display(Markdown(prog.ToLatex()))


def diditwork(solution: MathematicalProgramResult, verbose=True):
    if solution.is_success():
        printer = YAY
        printer("solve successful!", verbose=verbose)
    else:
        printer = ERROR
        printer("solve failed", verbose=verbose)
    printer(solution.get_optimal_cost(), verbose=verbose)
    printer(solution.get_solution_result(), verbose=verbose)
    printer("Solver is", solution.get_solver_id().name(), verbose=verbose)
    details = solution.get_solver_details()  # type: MosekSolverDetails
    if isinstance(details, MosekSolverDetails):
        printer(details, verbose=verbose)
        printer("time", details.optimizer_time, verbose=verbose)
        printer("rescode", details.rescode, verbose=verbose)
        printer("solution_status", details.solution_status, verbose=verbose)
    return solution.is_success()


def all_possible_combinations_of_items(item_set: T.List[str], num_items: int):
    """
    Recursively generate a set of all possible ordered strings of items of length num_items.
    """
    if num_items == 0:
        return [""]
    result = []
    possible_n_1 = all_possible_combinations_of_items(item_set, num_items - 1)
    for item in item_set:
        result += [item + x for x in possible_n_1]
    return result


def integrate_a_polynomial_on_a_box(
    poly: Polynomial, x: Variables, lb: npt.NDArray, ub: npt.NDArray
):
    assert len(lb) == len(ub)
    # compute by integrating each monomial term
    monomial_to_coef_map = poly.monomial_to_coefficient_map()
    expectation = Expression(0)
    for monomial in monomial_to_coef_map.keys():
        coef = monomial_to_coef_map[monomial]
        poly = Polynomial(monomial)
        for i in range(len(x)):
            x_min, x_max, x_val = lb[i], ub[i], x[i]
            integral_of_poly = poly.Integrate(x_val)
            poly = integral_of_poly.EvaluatePartial(
                {x_val: x_max}
            ) - integral_of_poly.EvaluatePartial({x_val: x_min})
        expectation += coef * poly.ToExpression()
    if not isinstance(expectation, float):
        ERROR("integral is not a value, it should be")
        ERROR(expectation)
        return None
    return expectation


class timeit:
    def __init__(self):
        self.times = []
        self.times.append(time.time())
        self.totals = 0
        self.a_start = None

    def dt(self, descriptor=None, print_stuff=True):
        self.times.append(time.time())
        if print_stuff:
            if descriptor is None:
                INFO("%.3fs since last time-check" % (self.times[-1] - self.times[-2]))
            else:
                descriptor = str(descriptor)
                INFO(descriptor + " took %.3fs" % (self.times[-1] - self.times[-2]))
        return self.times[-1] - self.times[-2]

    def T(self, descriptor=None):
        self.times.append(time.time())
        if descriptor is None:
            INFO("%.3fs since the start" % (self.times[-1] - self.times[0]))
        else:
            INFO(
                descriptor
                + " took %.3fs since the start" % (self.times[-1] - self.times[0])
            )

    def start(self):
        self.a_start = time.time()

    def end(self):
        self.totals += time.time() - self.a_start
        self.a_start = None

    def total(self, descriptor=None):
        INFO("All " + descriptor + " took %.3fs" % (self.totals))


def ChebyshevCenter(poly: HPolyhedron) -> T.Tuple[bool, npt.NDArray, float]:
    # Ax <= b
    m = poly.A().shape[0]
    n = poly.A().shape[1]

    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(n, "x")
    r = prog.NewContinuousVariables(1, "r")
    prog.AddLinearCost(np.array([-1]), 0, r)

    big_num = 100000

    prog.AddBoundingBoxConstraint(0, big_num, r)

    a = np.zeros((1, n + 1))
    for i in range(m):
        a[0, 0] = np.linalg.norm(poly.A()[i, :])
        a[0, 1:] = poly.A()[i, :]
        prog.AddLinearConstraint(
            a, -np.array([big_num]), np.array([poly.b()[i]]), np.append(r, x)
        )

    result = Solve(prog)
    if not result.is_success():
        return False, None, None
    else:
        return True, result.GetSolution(x), result.GetSolution(r)[0]


def offset_hpoly_inwards(hpoly: HPolyhedron, eps: float = 1e-5) -> HPolyhedron:
    A, b = hpoly.A(), hpoly.b()
    return HPolyhedron(A, b - eps)


def have_full_dimensional_intersection(
    hpoly1: HPolyhedron, hpoly2: HPolyhedron
) -> bool:
    intersection = hpoly1.Intersection(hpoly2)
    inward_intersection = offset_hpoly_inwards(intersection)
    return not inward_intersection.IsEmpty()

import typing as T

import numpy as np
import numpy.typing as npt

from util import timeit, INFO, YAY, ERROR, WARN  # pylint: disable=unused-import

from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    HPolyhedron,
)

FREE_POLY = "free_poly"
PSD_POLY = "psd_poly"

class ProgramOptions:
    def __init__(self):
        # -----------------------------------------------------------------------------------
        # general settings pertaining to any optimization program
        # -----------------------------------------------------------------------------------
        self.state_dim = 2

        self.verbose = True  # whether to scream debug stuff

        # -----------------------------------------------------------------------------------
        # OG GCS specific settings -- computing optimal solution from specific point.
        # -----------------------------------------------------------------------------------
        # initial state -- for regular GCS
        self.x0 = None  # type: npt.NDArray
        # solve MICP or convex relaxation
        self.use_convex_relaxation = False  # type: bool
        self.max_rounding_trials = 100  # type: int
        self.preprocessing = True  # type: bool

        # -----------------------------------------------------------------------------------
        # Polynomial GCS settings -- computing cost-to-go lower bound
        # -----------------------------------------------------------------------------------
        # degree of the potential per vertex
        self.potential_poly_deg = 2  # type: int
        # FREE_POLY -- potential is a free; PSD_POLY -- potential is PSD
        self.pot_type = PSD_POLY  # type: str
        # degree of the SOS polynomial edge constraint
        self.max_constraint_degree = 4  # type: int
        # whether to add more constraints by multiplying inequalities through
        self.putinar = True  # type: bool
        # push up potential at points
        self.push_up_points = []  # type: T.List[T.Tuple[str, npt.NDArray]]
        # push up potential at boxes
        self.push_up_boxes = []  # type: T.List[T.Tuple[str, npt.NDArray, npt.NDArray]]
        # instead of pushing up with the expected value, push up with a uniform sample of points
        self.push_up_box_at_uniform_sample_of_points = False  # type: bool
        self.num_uniform_push_up_sample_points = 31  # type: int

        # mosek solver details
        self.solver_debug_mode = False
        # feasability / tolerance gaps
        self.mosek_tolerance_gap = 1e-9
        self.mosek_primal_feas_gap = 1e-9
        self.mosek_dual_feas_gap = 1e-9

        # -----------------------------------------------------------------------------------
        # Control policy settings -- computing cost-to-go lower bound
        # -----------------------------------------------------------------------------------
        # False -- just minimize cost, True -- minimize delta to current potential
        # this is between edges
        self.complimentary_slackness_policy = False
        # one or two step lookahead policy
        self.lookahead = 1

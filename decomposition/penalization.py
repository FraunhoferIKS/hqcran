""" 
Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).

Contact: nicola.franco@iks.fraunhofer.de

"""

from copy import deepcopy

import numpy as np
from typing import Optional, cast, Union, Tuple, List

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import CplexOptimizer, OptimizationResult
from qiskit_optimization.converters import QuadraticProgramToQubo, QuadraticProgramConverter
from qiskit_optimization.converters import LinearInequalityToPenalty, LinearEqualityToPenalty, MaximizeToMinimize
from qiskit_optimization.problems import QuadraticObjective, Variable, Constraint
from qiskit_optimization.exceptions import QiskitOptimizationError


class ApproximateQuadraticProgram(QuadraticProgram):
    '''
    sub-class of Qiskits QuadraticProgram class:

    features:
    - checking the convertibility of the QP to a QUBO
    - approximating the constraints of a real-valued QP to make it convertible to a QUBO (different rounding types)
    - checking whether the original constraints are still fulfilled
    '''
    def __init__(self, quadratic_program: QuadraticProgram) -> None:
        # save original QP
        self._original_program = deepcopy(quadratic_program)

        # copy properties of the original QP to the approximate QP
        qp = deepcopy(quadratic_program)

        self._name = qp.name
        self._status = qp.status

        self._variables = qp.variables
        self._variables_index = qp.variables_index

        self._linear_constraints = qp.linear_constraints
        self._linear_constraints_index = qp.linear_constraints_index

        self._quadratic_constraints = qp.quadratic_constraints
        self._quadratic_constraints_index = qp.quadratic_constraints_index

        self._objective = qp.objective
        
        # specify two new instance variables (convertibility & factor)
        self._convertible2qubo = self._check_convertible2qubo() 
        self._factor = 1

    @property
    def original_program(self) -> QuadraticProgram:
        return self._original_program

    @property
    def convertible2qubo(self) -> bool:
        return self._convertible2qubo

    @property
    def factor(self) -> int:
        return self._factor

    def cplex_result(self) -> OptimizationResult:
        res = CplexOptimizer().solve(self)
        return res

    def to_unbalanced_penalization_qubo(self, penalty0=5, penalty1=1, penalty2=2) -> QuadraticProgram:
        return UnbalancedPenalizationQubo(penalty0=penalty0, penalty1=penalty1, penalty2=penalty2).convert(self)

    def convert2qubo(self) -> QuadraticProgram:
        if self._convertible2qubo:
            return QuadraticProgramToQubo().convert(self)
        raise ValueError("The current QP is not convertible to a QUBO due to float numbers inside the constraints. You can use 'self.real2int' to approximate the constraints")

    def real2int(self, n_decimals: int, rounding_mode: str = "standard") -> "ApproximateQuadraticProgram":
        '''
        convert real-valued constraints to integer-valued constraints by factorizing everything with the factor '10**n_decimals' and round up/down depending on the rounding mode.
        rounding modes: "standard", "improved", "contrary"
        '''
        self._check_n_decimals(n_decimals)
        self._check_rounding_mode(rounding_mode)

        nqp = ApproximateQuadraticProgram(self)

        nqp._factor = 10**n_decimals

        # linear constraints
        for nc,oc in zip(nqp._linear_constraints, self._original_program.linear_constraints):
            nc.linear = self._round2int(nqp._factor*oc.linear.to_array(), oc.sense.label, rounding_mode)
            if rounding_mode == "contrary":
                nc.rhs = - self._round2int(-nqp._factor*oc.rhs, oc.sense.label, rounding_mode) # round rhs to different direction than lhs
            else:
                nc.rhs = self._round2int(nqp._factor*oc.rhs, oc.sense.label, rounding_mode) # round rhs to same direction as lhs

        # quadratic constraints
        # for nc,oc in zip(self._quadratic_constraints, self._original_program.quadratic_constraints):
        #     nc.quadratic = self._round2int(self._factor*oc.quadratic.to_array(), oc.sense.label, rounding_mode)
        #     nc.linear = self._round2int(self._factor*oc.linear.to_array(), oc.sense.label, rounding_mode)
        #     nc.rhs = - self._round2int(-self._factor*oc.rhs, oc.sense.label, rounding_mode)

        nqp._convertible2qubo = nqp._check_convertible2qubo()

        return nqp

    def check_original_constraints(self, result: OptimizationResult, print_res: bool = True) -> None:
        # linear constraints
        for lc in self._original_program.linear_constraints:
            con = f"{lc.linear.to_array() @ result.x}{lc.sense.label}{lc.rhs}"
            if not eval(con):
                if print_res:
                    print(f"The original constraint '{lc.name}' is violated: {con}")
                return False
        
        # quadratic constraints: aren't supported so far
        if print_res:
            print(f"All original constraints are fulfilled.")
        return True

    def _check_convertible2qubo(self) -> bool:
        try:
            QuadraticProgramToQubo().convert(self)
            return True 
        except:
            return False

    def _check_rounding_mode(self, mode: str) -> None:
        if not mode in ["standard", "improved", "contrary"]:
            raise ValueError("'rounding_mode' has to be 'standard', 'improved' or 'contrary'")

    def _check_n_decimals(self, n_decimals) -> None:
        if not (isinstance(n_decimals, int) and n_decimals >= 0):
            raise ValueError("'n_decimals' has to be an integer greater than zero")

    def _round2int(self, array: np.ndarray or int, sense: str, rounding_mode: str) -> np.ndarray or int:
        if rounding_mode != "standard":
            if sense in ["<=", "<"]:
                return np.ceil(array)
            elif sense in [">=", ">"]:
                return np.floor(array)
        return np.rint(array)



class UnbalancedPenalizationQubo(QuadraticProgramConverter):

    def __init__(self, penalty0: Optional[float] = None, penalty1: Optional[float] = None, penalty2: Optional[float] = None) -> None:
        penalty1, penalty2 = _check_penalties(penalty1, penalty2)
        self._converters = [
            LinearInequalityToPenalty(penalty=penalty0),
            LinearInequalityToUnbalancedPenalization(penalty1=penalty1, penalty2=penalty2),
            LinearEqualityToPenalty(penalty=penalty0),
            MaximizeToMinimize(),
        ]

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        for conv in self._converters:
            problem = conv.convert(problem)
        return problem

    def interpret(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        for conv in self._converters[::-1]:
            x = conv.interpret(x)
        return cast(np.ndarray, x)


class LinearInequalityToUnbalancedPenalization(QuadraticProgramConverter):

    def __init__(self, penalty1: Optional[float] = 1, penalty2: Optional[float] = 2) -> None:
        self._src_num_vars: Optional[int] = None
        self._penalty1 = penalty1
        self._penalty2 = penalty2

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        
        # create empty QuadraticProgram model
        self._src_num_vars = problem.get_num_vars()
        dst = QuadraticProgram(name=problem.name)

        # set variables
        for x in problem.variables:
            if x.vartype == Variable.Type.CONTINUOUS:
                dst.continuous_var(x.lowerbound, x.upperbound, x.name)
            elif x.vartype == Variable.Type.BINARY:
                dst.binary_var(x.name)
            elif x.vartype == Variable.Type.INTEGER:
                dst.integer_var(x.lowerbound, x.upperbound, x.name)
            else:
                raise QiskitOptimizationError(f"Unsupported vartype: {x.vartype}")

        #TODO: If no penalties are given, set the penalty coefficients automatically

        # get original objective terms
        offset = problem.objective.constant
        linear = problem.objective.linear.to_dict()
        quadratic = problem.objective.quadratic.to_dict()
        sense = problem.objective.sense.value

        # convert linear inequality constraints into penalty terms
        for constraint in problem.linear_constraints:

            # adopt equality constraints from the original problem and tackle only inequality constraints
            if not constraint.sense != Constraint.Sense.EQ:
                dst.linear_constraint(
                    constraint.linear.coefficients,
                    constraint.sense,
                    constraint.rhs,
                    constraint.name,
                )
                continue
            
            else:
                constant = constraint.rhs
                row = constraint.linear.to_dict()
                sign = 1 if constraint.sense in [">=", ">"] else -1

                # constant parts of penalty
                offset += sense * (self._penalty2 * constant + self._penalty1 * sign) * constant

                # linear parts of penalty
                for j, coef in row.items():
                    linear[j] = linear.get(j, 0.0) - sense * (2 * self._penalty2 * constant + self._penalty1 * sign) * coef

                # quadratic parts of penalty
                for j, coef1 in row.items():
                    for k, coef2 in row.items():
                        tup = cast(Union[Tuple[int, int], Tuple[str, str]], (j, k))
                        quadratic[tup] = quadratic.get(tup, 0.0) + sense * self._penalty2 * coef1 * coef2
        
        if problem.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            dst.minimize(offset, linear, quadratic)
        else:
            dst.maximize(offset, linear, quadratic)

        # TODO: Update penalties to the ones just used

        return dst

    def interpret(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        if len(x) != self._src_num_vars:
            raise QiskitOptimizationError(
                "The number of variables in the passed result differs from "
                "that of the original problem."
            )
        return np.asarray(x)


def _check_penalties(self, penalty1 = None, penalty2 = None):
    if penalty1 == None:
        penalty1 = 1
    if penalty2 == None:
        penalty2 = 1
    return penalty1, penalty2




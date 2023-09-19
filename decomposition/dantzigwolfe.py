""" 
Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).

Contact: nicola.franco@iks.fraunhofer.de

"""


import time
import numpy as np
from qiskit_optimization.algorithms import CplexOptimizer, MinimumEigenOptimizer
from qiskit.algorithms.optimizers import SPSA, COBYLA
from docplex.mp.model import Model
import docplex
from dwave.system import AutoEmbeddingComposite
from dwave import embedding
import dimod
from parameters import Param, DWData
from qiskit.algorithms.minimum_eigen_solvers import QAOA, VQE
from qiskit.primitives import Sampler
from qiskit_optimization.translators import from_docplex_mp
from qiskit.utils.algorithm_globals import algorithm_globals

class Matrices:
    """ 
    Matrices of the problem

    Parameters
    ----------
    A: numpy.ndarray
        Matrix of the real variables
    B: numpy.ndarray
        Matrix of the binary variables
    C: numpy.ndarray
        Matrix of the real variables
    b: numpy.ndarray
        Vector of the real variables
    d: numpy.ndarray
        Vector of the binary variables
    g: numpy.ndarray
        Vector of the real variables
    """
    def __init__(self, matrices) -> None:
        self.A, self.B, self.C, self.b, self.d, self.g = matrices


class DWDecomposition:


    def __init__(self, parameters: Param, matrices: list, bounds: tuple):
        pass
        self.parameters = parameters
        # self.num_classes = 10
        self.matrices = Matrices(matrices)
        self.bounds = bounds
        
        # self.exact_optimizer = CplexOptimizer()
        self.data = DWData()
        self.master_objective = None
        self.sub_real = 0
        self.sub_bin = 0
        self.y_master = []
        self.alpha = []
        self.eta = 0
        self.xi = 0

        self.gamma = []

        self.init_convex()
        self.initial_extreme_points = self.starting_point()
        
        self.z_solution = {
            'extreme point': [self.initial_extreme_points['z']],
            'extreme ray': []
            } 
        self.y_solution = {
            'extreme point': [self.initial_extreme_points['y']],
            'extreme ray': []
        }
    

    def run(self, adversary: int):

        init_time = time.time()
        matrices = self.matrices

        matrices.g[int(adversary - 10)] = -1
        
        data = DWData()
        data.adversary = adversary
        data.num_qubits[0] = matrices.B.shape[1]

        sub_real = Model()

        z = sub_real.continuous_var_list(
            matrices.A.shape[1], lb=self.bounds[0], ub=self.bounds[1], name='z'
            )
        sub_real.add_constraints_(
            sub_real.scal_prod(z, matrices.C[i, :]) >= matrices.d[i] 
            for i in range(matrices.C.shape[0])
            )

        convex_objective = self.solve_convex(matrices)

        if convex_objective > 0:
            data.dual_master_obj.append(convex_objective)
        else:

            self.eta, self.xi = 0, 0
            
            for step in range(self.parameters.steps):

                """ Subproblem real """
                sub_real.minimize((matrices.g - self.alpha@matrices.A)@z)
                sub_real_solution = sub_real.solve()

                sub_real_obj = sub_real_solution.get_objective_value()
                z_sol = sub_real_solution.get_values(z)

                """ Subproblem binary """
                linear = -np.array(self.alpha)@matrices.B

                y_sol, sub_bin_obj = self.compute_pricing_binary(
                    matrices=matrices, 
                    linear=linear,
                    step=step,
                    )
                

                if sub_real_obj < self.eta:
                    if z_sol not in self.z_solution['extreme point']:
                        self.z_solution['extreme point'].append(z_sol)
                    # print('z extreme point')
                elif z_sol not in self.z_solution['extreme ray']:
                    self.z_solution['extreme ray'].append(z_sol)
                    # print('z extreme ray')


                if sub_bin_obj < self.xi and \
                    y_sol not in self.y_solution['extreme point']:
                    self.y_solution['extreme point'].append(y_sol)
                    # print(f'y extreme point: {y_sol}')
                elif y_sol not in self.y_solution['extreme ray']:
                    self.y_solution['extreme ray'].append(y_sol)
                    # print(f'y extreme ray: {y_sol}')

                """ Master problem """ 
                self.master_objective = self.compute_master_convex(
                    matrices=matrices,
                    # z_sol=self.z_solution,
                    # y_sol=self.y_solution,
                    y_branches=None
                )

                if self.parameters.verbose:

                    print("Master Convex", self.master_objective, self.y_master)
                    print("Sub Objectives:", sub_real_obj, self.eta, sub_bin_obj, self.xi)
                    # print("Master Binary", self.master_objective, self.y_master)                
                # print("alpha", self.alpha)
                dual_master_obj = self.alpha@matrices.b + self.eta + self.xi #- sub_real_obj - sub_bin_obj
                dual_master_obj = -dual_master_obj

                # Print results
                if self.parameters.verbose:
                    print(f"Master: {self.master_objective}, {self.y_master}")
                    print(f"Dual master: {dual_master_obj}")
                    print(f"Sub real: {sub_real_obj}")
                    print(f"Sub binary: {sub_bin_obj}, {y_sol}")


                # Store data
                if step==0 or dual_master_obj > data.dual_master_obj[-1]:
                    data.dual_master_obj.append(dual_master_obj)
                    data.sub_objectives.append(dual_master_obj)
                else:
                    data.dual_master_obj.append(data.dual_master_obj[-1])
                    data.sub_objectives.append(dual_master_obj)

                data.sub_real_obj.append(sub_real_obj)
                data.sub_bin_obj.append(sub_bin_obj)
                data.master_objectives.append(self.master_objective)
                data.gap = np.abs(self.master_objective - data.dual_master_obj[-1])

                if self.master_objective < 0:
                    if self.parameters.verbose:
                        print("Master is negative -> Counter example")
                    break
                elif dual_master_obj > 0:
                    if self.parameters.verbose:
                        print("Dual is positive -> No counter example")
                        print("dual positive:", dual_master_obj)
                    break
                elif data.gap < self.parameters.objectives_gap:
                    is_correct, objective = self.verify(matrices, y_sol)
                    if self.parameters.verbose:
                        print("Master Obj:", y_sol, self.master_objective)
                        print(f"{is_correct} solution found")
                        print("Objective:", objective)
                        print("Gap:", data.gap)
                    break
        
        if self.master_objective is not None and self.master_objective < 0:
            data.solution = self.master_objective
        else:
            data.solution = data.dual_master_obj[-1]
        data.total_time = time.time() - init_time

        matrices.g[int(adversary - 10)] = 0

        return data
    

    def compute_pricing_binary(self, matrices, linear, step):#, y_branches):

        if any(linear):
            if self.parameters.solver == "cplex":
                
                sub_bin = Model()
                y = sub_bin.binary_var_list(matrices.B.shape[1], name='y')
                sub_bin.minimize(sub_bin.scal_prod(y, linear))
            

                sub_bin_solution = sub_bin.solve()
                y_sol = sub_bin_solution.get_values(y)
                sub_bin_obj = sub_bin_solution.get_objective_value()

            elif self.parameters.solver in 'ibm_simulator':
                
                sub_bin = Model()
                y = sub_bin.binary_var_list(matrices.B.shape[1], name='y')
                sub_bin.minimize(sub_bin.scal_prod(y, linear))                    
                
                problem = from_docplex_mp(sub_bin)
                algorithm_globals.random_seed = 12345
        
                qaoa_mes = QAOA(
                    quantum_instance=self.parameters.quantum_solver,
                    optimizer=COBYLA(), reps=5
                    )
                algorithm = MinimumEigenOptimizer(qaoa_mes)
                result = algorithm.solve(problem)
                y_sol = list(result.x)
                sub_bin_obj = result.fval

            elif self.parameters.solver in 'vqe':

                sub_bin = Model()
                y = sub_bin.binary_var_list(matrices.B.shape[1], name='y')
                sub_bin.minimize(sub_bin.scal_prod(y, linear)) 

                problem = from_docplex_mp(sub_bin)
                algorithm_globals.random_seed = 12345

                vqe = VQE(
                    quantum_instance=self.parameters.quantum_solver,
                    optimizer=COBYLA()
                    )

                algorithm = MinimumEigenOptimizer(vqe)
                result = algorithm.solve(problem)
                y_sol = list(result.x)
                sub_bin_obj = result.fval

            elif self.parameters.solver in 'dwave_hybrid | dwave_qpu | dwave_simulator':
                
                bqm = dimod.BQM(np.diag(linear), vartype='BINARY')

                if self.parameters.solver == 'dwave_qpu':
                    sub_bin_solution = AutoEmbeddingComposite(
                        self.parameters.quantum_solver).sample(
                        bqm, #chain_strength=embedding.chain_strength.scaled, 
                        label="Step {0:d}".format(step), 
                        num_reads=self.parameters.dwave_reads
                    )
                else:
                    sub_bin_solution = self.parameters.quantum_solver.sample(
                        bqm, 
                        label="Step {0:d}".format(step), 
                        num_reads=self.parameters.dwave_reads, 
                        num_sweeps=self.parameters.dwave_reads*500
                        )
                
                y_sol = np.fromiter(sub_bin_solution.first.sample.values(), dtype=int).tolist()
                sub_bin_obj = y_sol@linear
        else:
            y_sol = matrices.B.shape[1]*[0]
            sub_bin_obj = 0

        return y_sol, sub_bin_obj
    
    def compute_master_binary(self, matrices):

        z_sol = self.z_solution['extreme point'] + self.z_solution['extreme ray']
        y_sol = self.y_solution['extreme point'] + self.y_solution['extreme ray']

        model = Model()
        lambda_ = model.binary_var_list(len(z_sol), name='lambda')
        mu_ = model.binary_var_list(len(y_sol), name='mu')

        A_tilde = np.array([
            [a_ij * lambda_[i] for a_ij in matrices.A@z_i]
            for i, z_i in enumerate(z_sol)
        ])

        B_tilde = np.array([
            [b_ij * mu_[i] for b_ij in matrices.B@y_i]
            for i, y_i in enumerate(y_sol)
        ])

        vector_of_index_alpha = []
        for i in range(len(matrices.b)):
            constr = sum(A_tilde[:, i]) + sum(B_tilde[:, i])
            if isinstance(constr, docplex.mp.linear.LinearExpr):
                model.add_constraint_(constr >= matrices.b[i])
                vector_of_index_alpha.append(True)
            else:
                vector_of_index_alpha.append(False)

        model.add_constraint_(
            model.sum([l for l in lambda_[:len(self.z_solution['extreme point'])]]) == 1)
        model.add_constraint(
            model.sum([m for m in mu_[:len(self.y_solution['extreme point'])]]) == 1)

        model.minimize(
            model.sum([
                lambda_[i]*(matrices.g@x_i) for i, x_i in enumerate(z_sol)])
        )
        
        solution = model.solve()

        try:
            lambda_sol = solution.get_values(lambda_)
            mu_sol = solution.get_values(mu_)

            x_sol = sum([lmbd * np.array(x_i) for lmbd, x_i in zip(lambda_sol, z_sol)])
            self.y_master = list(sum([mu * np.array(y_i) for mu, y_i in zip(mu_sol, y_sol)]))
            
            self.master_objective = solution.get_objective_value()
        
        except:
            if self.parameters.verbose: 
                print(model.get_solve_details())
            self.master_objective = None


    def compute_master_convex(self, matrices, y_branches:dict):
        
        z_sol = self.z_solution['extreme point'] + self.z_solution['extreme ray']
        y_sol = self.y_solution['extreme point'] + self.y_solution['extreme ray']
        
        model = Model()
        lambda_ = model.continuous_var_list(len(z_sol), name='lambda')
        mu_ = model.continuous_var_list(len(y_sol), name='mu')
        # mu_ = model.binary_var_list(len(y_sol), name='mu')

        A_tilde = np.array([
            [a_ij * lambda_[i] for a_ij in matrices.A@z_i]
            for i, z_i in enumerate(z_sol)
        ])

        B_tilde = np.array([
            [b_ij * mu_[i] for b_ij in matrices.B@y_i]
            for i, y_i in enumerate(y_sol)
        ])

        vector_of_index_alpha = []
        for i in range(len(matrices.b)):
            constr = sum(A_tilde[:, i]) + sum(B_tilde[:, i])
            if isinstance(constr, docplex.mp.linear.LinearExpr):
                model.add_constraint_(constr >= matrices.b[i])
                vector_of_index_alpha.append(True)
            else:
                vector_of_index_alpha.append(False)

        if y_branches:
            for pos, up_low in y_branches:
                model.add_constraint_(
                        model.sum([
                            mu_[i] if y_sol[i][pos] == up_low else 0
                            for i in range(len(y_sol))
                            ])
                        == 1
                    )


        model.add_constraint_(
            model.sum([l for l in lambda_[:len(self.z_solution['extreme point'])]]) == 1)
        model.add_constraint(
            model.sum([m for m in mu_[:len(self.y_solution['extreme point'])]]) == 1)

        model.minimize(
            model.sum([
                lambda_[i]*(matrices.g@x_i) for i, x_i in enumerate(z_sol)
            ])
        )
        
        solution = model.solve()

        try:
            lambda_sol = solution.get_values(lambda_)
            mu_sol = solution.get_values(mu_)

            x_sol = sum([lmbd * np.array(x_i) for lmbd, x_i in zip(lambda_sol, z_sol)])
            self.y_master = list(sum([mu * np.array(y_i) for mu, y_i in zip(mu_sol, y_sol)]))
            
            dual_variables = model.cplex.solution.get_dual_values()

            length_alpha = sum(vector_of_index_alpha)
            alpha_sol = dual_variables[:length_alpha]

            dual_var_length = len(alpha_sol) #+ len(beta_sol)

            gamma_sol = np.zeros(matrices.B.shape[1])
            
            self.gamma = list(gamma_sol)


            self.eta = dual_variables[-2]
            self.xi = dual_variables[-1]

            j = 0
            for i, idx in enumerate(vector_of_index_alpha):
                if idx:
                    vector_of_index_alpha[i] = alpha_sol[j]
                    j +=1
                else:
                    vector_of_index_alpha[i] = self.parameters.sub_bound

            alpha_sol = vector_of_index_alpha
            self.alpha = alpha_sol


            return solution.get_objective_value()
            
        
        except:
            if self.parameters.verbose: 
                print(model.get_solve_details())
            self.master_objective = None
            # return False, None, None
        



    def verify(self, matrices, y_sol) -> list:

            model = docplex.mp.model.Model(name="primal")
            z = model.continuous_var_list(matrices.A.shape[1], lb=-model.infinity)

            model.minimize(model.scal_prod(z, matrices.g))
            model.add_constraints_( 
                model.scal_prod(z, matrices.A[i, :]) + matrices.B[i, :]@y_sol 
                >= matrices.b[i] for i in range(matrices.A.shape[0])
            )
            model.add_constraints_(
                model.scal_prod(z, matrices.C[i, :]) >= matrices.d[i]
                for i in range(matrices.C.shape[0])
            )

            solution = model.solve()

            if solution:
                return True, solution.get_objective_value()
            else:
                return False, None

    def solve_convex(self, matrices: Matrices):
        
        self.convex.minimize(self.convex.scal_prod(self.convex_z, matrices.g))

        solution = self.convex.solve()
        
        dual_variables = self.convex.cplex.solution.get_dual_values()
        self.alpha = dual_variables[:matrices.A.shape[0]]

        return solution.get_objective_value()

    def init_convex(self):

        self.convex = Model()

        self.convex_z = self.convex.continuous_var_list(
            self.matrices.A.shape[1], lb=-self.convex.infinity, name='z'
            )
        self.convex_y = self.convex.continuous_var_list(
            self.matrices.B.shape[1], lb=0, ub=1, name='y'
        )

        self.convex.add_constraints_(
            self.convex.scal_prod(self.convex_z, self.matrices.A[i, :]) + 
            self.convex.scal_prod(self.convex_y, self.matrices.B[i, :]) >= 
            self.matrices.b[i]
            for i in range(self.matrices.A.shape[0])
            )
        
        self.convex.add_constraints_(
            self.convex.scal_prod(self.convex_z, self.matrices.C[i, :]) >= self.matrices.d[i]
            for i in range(self.matrices.C.shape[0])
            )

    def starting_point(self):
        

        model = Model(name="primal")
        z = model.continuous_var_list(self.matrices.A.shape[1], lb=-model.infinity)
        y = model.binary_var_list(self.matrices.B.shape[1])

        model.add_constraints_(
            model.scal_prod(z, self.matrices.A[i, :]) + 
            model.scal_prod(y, self.matrices.B[i, :]) >= 
            self.matrices.b[i]
            for i in range(self.matrices.A.shape[0])
            )
        
        model.add_constraints_(
             model.scal_prod(z, self.matrices.C[i, :]) >= self.matrices.d[i]
            for i in range(self.matrices.C.shape[0])
            )
        
        model.minimize(0)
        solution = model.solve()

        return {'z': solution.get_values(z), 'y': solution.get_values(y)}
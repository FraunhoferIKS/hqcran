""" 
Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).

Contact: nicola.franco@iks.fraunhofer.de

"""

import time
import numpy as np
# Gurobi
import gurobipy as gp
from gurobipy import GRB
# Qiskit
from qiskit_optimization.algorithms import CplexOptimizer, MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.runtime import QAOAClient
from qiskit.algorithms.optimizers import SPSA
from qiskit.algorithms import QAOA
# Classic
from docplex.mp.model import Model
# D-Wave
import dwave
from dwave import embedding
from dwave.system import AutoEmbeddingComposite, DWaveSampler
from tabu import TabuSampler
import dimod
# import hreduction
# from hreduction.rancic import rancic_qubo
from qiskit_optimization.translators import from_docplex_mp
from .penalization import UnbalancedPenalizationQubo, ApproximateQuadraticProgram
from parameters import Param, Data

class BendersDecomposition:
    def __init__(self, parameters: Param, matrices: list):
        
        """ 
        Initialize the problem with the given constraints and parameters 
        
        :param parameters: optimization.Param  
        :param constraints: constraints.ConstraintsGenerator
        """
        # Initialize parameters & constraints
        self.param = parameters
        self.n_classes = 10 #TODO: from network
        self.A, self.B, self.C, self.b, self.d, self.g = matrices
        self.n_binaries = self.B.shape[1]

        self.sub_model, self.sub_vars = self.sub_initialization() # Initialize the sub
        self.verifier, self.z = self.verifier_initialization()
        if self.param.real_var:
            self.granularity_p = np.array([1], dtype=np.float32)
        else:    
            # Fixed-point approximation
            self.granularity_p = np.concatenate(([-self.param.weight_p * np.power(2, self.param.num_qubits_p - 1)],
                                [self.param.weight_p * np.power(2, i) for i in range(self.param.num_qubits_p - 1)]))
        
        self.set_of_cuts, self.num_qubits_a = [], [] # Initialize the set of cuts
        self.master_variables = []
        self.core_y = np.ones(self.n_binaries)
        self.last_minimum = np.zeros(self.n_binaries)

        self.exact_optimizer = CplexOptimizer() # Initialize the classical solver
        self.data = Data() # Initialize timer, objectives and info

        # Rancic specific caching of the previous solution
        self.rancic_prev_assignment = None

        self.unbalanced_penalities = parameters.unbalanced_penalities

    def update_penalities(self, penalities:list):
        """
            Lambda 0, Lambda 1, Lambda 2
        """
        self.unbalanced_penalities = penalities

    
    
    def master_initialization(self):
        """
            Master problem initialization: define problem variables p  and y

        :return (model, p, y): (Docplex Model, Docplex Vars, Docplex Vars)
        """
        self.data = Data()
        self.set_of_cuts.clear()
        self.num_qubits_a.clear()
        self.master_variables.clear()
        self.core_y = np.ones(self.n_binaries)

        if self.param.real_var:
            if self.param.gurobi:
                model = gp.Model("Master")
                model.Params.LogToConsole = self.param.verbose
                model.Params.Threads = self.param.threads
                p = model.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name='p')
            else:
                model = Model("Master")
                p = model.continuous_var(lb=-np.inf, name="p")
        else:
            model = QuadraticProgram("Master")
            p = model.binary_var_list(self.param.num_qubits_p, name="p")

        if self.param.gurobi:
            y = model.addVars(self.n_binaries, vtype=GRB.BINARY)
        else:
            y = model.binary_var_list(self.n_binaries, name="y")

        return model, p, y

    def sub_initialization(self):

        """
            Sub problem initialization: set the boundaries of alpha & beta and the constraint:
                alpha*A + beta*C = gT

        :return (model, var): Cplex or Gurobi Model and Vars
        """

        # alpha and beta Bounds
        bounds = [self.param.sub_bound if np.any(row) else np.inf for row in self.B]
        bounds += [self.param.sub_bound if elem else np.inf for elem in self.d]
        # Constraints
        constr = np.transpose(np.concatenate((self.A[:,:-self.n_classes], self.C[:,:-self.n_classes])))

        if self.param.gurobi:

            # Initialize the sub
            model = gp.Model("Sub")
            model.Params.LogToConsole = self.param.verbose
            model.Params.Threads = self.param.threads
            
            var = model.addMVar(len(bounds), lb=0, ub=bounds, vtype=GRB.CONTINUOUS)
            
            # Sub problem constraints up to the last layer
            model.addMConstr(constr, var, '=', self.g[:-self.n_classes])

        else:
            # Initialize the sub
            model = Model("Sub")
            model.parameters.threads = self.param.threads

            var = model.continuous_var_list(len(bounds), lb=0, ub=bounds)

            # Sub problem constraints up to the last layer
            model.add_constraints_(
                model.scal_prod(var, constr[i, :]) == self.g[i] 
                for i in range(self.A.shape[1]-self.n_classes)
                )

        return model, var


    def verifier_initialization(self):
        """
            Initialize the verifier: Set the constraints to compute Eq.10 for a given y

            :return (model, z): (Docplex Model, Docplex Vars)
        """
        model = Model("Verifier")
        model.parameters.threads = self.param.threads
        z = model.continuous_var_list(self.C.shape[1], lb=-model.infinity, name="z")
        model.add_constraints_(model.scal_prod(z, self.C[i, :]) >= self.d[i] 
            for i in range(self.C.shape[0]))

        return model, z

    
    def run(self, adversary:int) -> Data:
        """  
            Main algorithm: Iterations between Master and Sub problem

        :param adversary: (int) adversary class to test
        :return data: (Param.Data) 
        """
        self.master, self.p, self.y = self.master_initialization()
        # Change the constraints to test the adversary
        self.g[int(adversary - self.n_classes)] = - 1
        # Set the constraints of the dual for the last layer
        constr = np.transpose(np.concatenate((self.A[:,-self.n_classes:], self.C[:,-self.n_classes:])))
        if self.param.gurobi:
            sub_constraints = self.sub_model.addMConstr(constr, self.sub_vars, '=', self.g[-self.n_classes:])
        else:
            sub_constraints = self.sub_model.add_constraints(
                self.sub_model.scal_prod(self.sub_vars, constr[i, :]) == self.g[len(self.g) - self.n_classes + i] 
                for i in range(self.n_classes)
                )

        if self.param.verbose: print("----------------------------------------- START \
            -----------------------------------------\n")
        
        start = time.perf_counter()
        for step in range(self.param.steps):

            if self.param.verbose:
                print(f"\n--------------- STEP {step} ----------------")
                print("------------ MASTER PROBLEM ------------\n")

            self.master_solver(step) if step > 0 else self.master_first_step()

            if self.param.verbose: print("\n------------- SUB PROBLEM -------------\n")
            self.sub_solver()
            
            # Check if the solution of the master objective value is close to the sub one
            if np.abs(self.data.master_objectives[-1] - self.data.sub_objectives[-1]) < self.param.objectives_gap or \
                self.data.master_objectives[-1] > self.data.sub_objectives[-1]:

                solution = self.verification() # Verify the correctness
                
                if solution:
                    if solution.get_objective_value() < self.data.master_objectives[-1]:
                        self.data.solution = solution.get_objective_value()
                    else:
                        self.data.solution = self.data.master_objectives[-1]
                    if self.param.verbose:
                        print("\n---------- Correct Solution ------------\n")
                    break
                else:
                    self.master_variables.pop()
                    self.data.master_objectives.pop()
                    if self.param.verbose:
                        print("\n---------- Wrong Solution -------------")
            elif self.param.real_var and (self.data.master_objectives[-1] > 0 or self.data.sub_objectives[-1] < 0):
                self.data.solution = self.data.master_objectives[-1]
                break

        self.data.total_time = time.perf_counter() - start

        self.data.total_time = time.perf_counter() - start

        if self.data.solution:
            self.data.gap = self.data.sub_objectives[-1] - self.data.solution
        else:
            self.data.gap = float('nan')

        self.g[int(adversary - 10)] = 0  # reset to zero the adversary
        if self.param.gurobi:
            self.sub_model.remove(sub_constraints)
        else:
            self.sub_model.remove_constraints(sub_constraints)

        if self.param.verbose:
            print("\n----------------------------------------- END -----------------------------------------\n")
            print("Solution: {0:.5f}".format(self.data.solution))
            print("Gap between the master and sub objectives: {0:.5f}".format(self.data.gap))
            print("Master Objective: {0:.5f}".format(self.data.master_objectives[-1]))
            print("Sub Objective: {0:.5f}".format(self.data.sub_objectives[-1]))
            print("Total Time: {0:.5f}".format(self.data.total_time))
            print("Master Total Time: {0:.5f}".format(np.sum(self.data.master_times)))
            print("Sub Total Time: {0:.5f}".format(np.sum(self.data.sub_times)))
            print("\n---------------------------------------\n")
        
        return self.data


    def master_first_step(self):
        """         
        Master first step optimization: the problem has no constraints
        
        :return [np.ndarray, float] = variables, objective        
        """

        linear = np.concatenate((self.granularity_p, np.zeros(self.n_binaries)), axis=0, dtype=np.float32)
        
        if self.param.real_var:
            variables = np.concatenate(([-np.inf], np.zeros(self.n_binaries)))
        else:
            variables = np.concatenate(([1], np.zeros(len(self.p) - 1 + self.n_binaries)))

        objective = linear @ variables

        self.data.master_objectives.append(objective)
        self.master_variables.append(variables)

        if self.param.verbose:
            print("Master Objective: {}".format(objective))
            print("Master Variables: {}".format(variables))


    def master_solver(self, step: int):
        """         
        Master Optimization 

        :param step: int      
        """
        # Generate the new optimality cuts
        self.generate_cut()

        # if self.param.qubo:

        #     quadratic, linear, constant = self.generate_quadratic(step)

        #     if self.param.verbose:
        #         print("Total number of qubits: {}, Number of qubits for the last cut: {}".format(
        #             quadratic.shape[0], self.num_qubits_a[-1]))

        # Universal quantum computing systems (IBM)
        if self.param.solver in 'cplex':
            
            if not self.param.real_var:

                # self.master.minimize(quadratic=quadratic, linear=linear, constant=constant)
                # start = time.perf_counter()
                # solution = self.param.quantum_solver.solve(self.master)
                # time_solver = time.perf_counter() - start
                # try:
                #     variables = solution.x
                #     objective = self.granularity_p@variables[:len(self.p)]*self.param.weight_t
                # except:
                #     raise NameError("Solution status: " + str(solution.status))
                # self.master.add_constraint_(
                #     self.master.scal_prod(self.y, self.set_of_cuts[-1][:-1]) + self.set_of_cuts[-1][-1] <= self.p
                #     )
                
                self.master.linear_constraint(
                    linear=np.concatenate((-self.granularity_p, self.set_of_cuts[-1][:-1])),
                    sense='<=', rhs=-self.set_of_cuts[-1][-1]
                )

                self.master.minimize(linear=np.concatenate((self.granularity_p, np.zeros(self.n_binaries))))
                # print(self.master.prettyprint())
                start = time.perf_counter()
                qubo = UnbalancedPenalizationQubo(
                    penalty0=self.unbalanced_penalities[0], 
                    penalty1=self.unbalanced_penalities[1], 
                    penalty2=self.unbalanced_penalities[2]).convert(self.master)
                # print(qubo.prettyprint())
                aqp = ApproximateQuadraticProgram(qubo)
                time_solver = time.perf_counter() - start
                solution = aqp.cplex_result()
                variables = solution.x
                objective = self.granularity_p@variables[:len(self.p)]*self.param.weight_t     

            else:
                if self.param.gurobi:
                    self.master.addConstr(
                        gp.quicksum([self.y[i]*self.set_of_cuts[-1][:-1][i] for i in range(self.n_binaries)]) 
                        + self.set_of_cuts[-1][-1] <= self.p
                    )

                    self.master.setObjective(self.p/self.param.weight_t, GRB.MINIMIZE)

                    self.master.optimize()
                    objective = self.master.ObjVal
                    time_solver = self.master.Runtime
                    variables = np.array([var.X for var in self.master.getVars()])
                else:
                    if self.param.qubo:
                        variables = [var for var in self.master.iter_variables()]
                        self.master.set_objective("min", variables@quadratic@variables + linear@variables + constant)
                        
                        solution = self.master.solve()
                        time_solver = self.master.solve_details.time
                        
                        try:
                            objective = self.p.solution_value*self.param.weight_t                
                            variables = np.array([solution.get_value(var) for var in self.master.iter_variables()])
                        except:
                            objective = self.data.master_objectives[-1]
                            variables = self.master_variables[-1]
                        
                    else:
                        self.master.add_constraint_(
                            self.master.scal_prod(self.y, self.set_of_cuts[-1][:-1]) + self.set_of_cuts[-1][-1] <= self.p
                            )

                        # if self.param.hamming:
                        #     hamming_distance = 0.5*np.sum(
                        #         [(y - y_t)**2 for y, y_t in zip(self.y, self.master_variables[-1][1:1+self.n_binaries])]) 
                        #     hamming = self.master.add_constraint(hamming_distance <= int(self.n_binaries/2))

                        self.master.set_objective("min", self.p/self.param.weight_t)
                        start = time.perf_counter()
                        solution = self.master.solve()
                        time_solver = time.perf_counter() - start

                        # start = time.perf_counter()
                        # quad_master = from_docplex_mp(self.master)
                        # qubo = UnbalancedPenalizationQubo(penalty0=1.0792, penalty1=0.9603, penalty2=0.0371).convert(quad_master)
                        # aqp = ApproximateQuadraticProgram(qubo)
                        # time_solver = time.perf_counter() - start
                        # solution = aqp.cplex_result()
                        # variables = solution.x
                        # objective = variables[0]*self.param.weight_t                
                    

                    try:
                        objective = self.p.solution_value*self.param.weight_t                
                        variables = np.array([solution.get_value(var) for var in self.master.iter_variables()])
                    except:
                        objective = self.data.master_objectives[-1]
                        variables = self.master_variables[-1]

                # if self.param.hamming and not self.param.qubo: 
                #     self.master.remove_constraint(hamming)

        elif self.param.solver in 'ibm_cloud':

            if quadratic.shape[0] < 27:
                backend = self.param.quantum_solver.get_backend('ibmq_toronto')
                # options = {'backend_name': 'ibmq_toronto'}
            elif quadratic.shape[0] < 65:
                backend = self.param.quantum_solver.get_backend('ibmq_brooklyn')
                # options = {'backend_name': 'ibmq_brooklyn'}
            else:
                backend = self.param.quantum_solver.get_backend('ibm_washington')
                # options = {'backend_name': 'ibm_washington'}

            # dictionary to store the history of the optimization
            history = {"nfevs": [], "params": [], "energy": [], "std": []}

            def store_history_and_forward(nfevs, params, energy, std):
                # store information
                history["nfevs"].append(nfevs)
                history["params"].append(params)
                history["energy"].append(energy)
                history["std"].append(std)

            qaoa = QAOAClient(optimizer=SPSA(maxiter=5), backend=backend, provider=self.param.quantum_solver,
                initial_point=np.zeros(4), callback=store_history_and_forward, reps=2, shots=1024)

            solver = MinimumEigenOptimizer(qaoa)

            self.master.minimize(quadratic=quadratic, linear=linear, constant=constant) # Define the QUBO problem

            start = time.perf_counter()
            solution = solver.solve(self.master)
            time_solver = time.perf_counter() - start
            print(history)
            try:
                variables = solution.x
                objective = self.granularity_p@variables[:len(self.p)]*self.param.weight_t
            except:
                raise NameError("Solution status: " + str(solution.status))

        # Quantum Annealing systems (D-WAVE)
        elif self.param.solver in 'dwave_hybrid | dwave_qpu | dwave_simulator':
            
            bqm = dimod.BQM(linear, quadratic, constant, vartype='BINARY') # Binary quadratic Model -> bqm

            if self.param.solver == 'dwave_hybrid':
                solution = self.param.quantum_solver.sample(bqm, label="Step {0:d}".format(step))
                time_solver = solution.info['run_time']/1e6
            
            elif self.param.solver == 'dwave_qpu':
                
                start = time.perf_counter()
                solution = AutoEmbeddingComposite(self.param.quantum_solver).sample(
                    bqm, chain_strength=embedding.chain_strength.scaled, 
                    label="Step {0:d}".format(step), num_reads=self.param.dwave_reads
                    )
                time_solver = time.perf_counter() - start
                
                if self.param.verbose:
                    print("Percentage of samples with high rates of breaks is {0:.4f}.".format(
                        np.count_nonzero(solution.record.chain_break_fraction > 0.05)/self.param.dwave_reads*100))
                
                    if self.param.dwave_inspector: dwave.inspector.show(solution)

            else:
                            
                start = time.perf_counter()
                solution = self.param.quantum_solver.sample(bqm, 
                    label="Step {0:d}".format(step), num_reads=self.param.dwave_reads, 
                    num_sweeps=self.param.dwave_reads*500)
                time_solver = time.perf_counter() - start

            variables = np.fromiter(solution.first.sample.values(), dtype=int)
            objective = self.granularity_p@variables[:len(self.p)]*self.param.weight_t

            if self.param.verbose:
                print(f"D-Wave solution: {objective}")
                print(f"D-Wave time: {time_solver} in seconds")
        
        # Store values
        self.data.master_times.append(time_solver)
        if not self.param.real_var: 
            self.data.num_qubits.append(len(solution.x))

        self.data.master_objectives.append(objective)
        self.master_variables.append(variables)

        if self.param.verbose:
            print(f"Master Objective: {objective}")
            print("Master Variables y: {}".format(
                variables[len(self.granularity_p):len(self.granularity_p)+self.n_binaries]))

    
    def sub_solver(self):
        
        """ 
            Sub problem optimization and independent Magnanti & Wong cut generation
        
        """
        y = self.master_variables[-1][len(self.granularity_p):len(self.granularity_p)+self.n_binaries]

        coeff = np.concatenate(((self.b - self.B@y), self.d))
        if self.param.gurobi:
            self.sub_model.setObjective(coeff@self.sub_vars, GRB.MAXIMIZE)
            # self.sub_model.setMObjective(None, c, 0.0, None, None, xc, GRB.MAXIMIZE)
    
            self.sub_model.optimize()
            objective = self.sub_model.ObjVal
            time = self.sub_model.Runtime        
        
        else:
            self.sub_model.set_objective("max", self.sub_model.scal_prod(self.sub_vars, coeff))
            
            solution = self.sub_model.solve()
            objective = solution.get_objective_value()
            time = self.sub_model.solve_details.time
    
        if self.param.verbose: print(f"Sub objective: {objective}")


        if len(self.data.sub_objectives) and objective > self.data.sub_objectives[-1]:
            self.data.sub_objectives.append(self.data.sub_objectives[-1])
        else:
            self.last_minimum = y
            self.data.sub_objectives.append(objective)



        if self.param.magnanti:
            # Generate Independent Pareto optimal cuts
            self.core_y = self.core_y/2 + y/2
            coeff = np.concatenate(((self.b - self.B@self.core_y), self.d))
            
            if self.param.gurobi:

                self.sub_model.setObjective(coeff@self.sub_vars, GRB.MAXIMIZE)
            
                self.sub_model.optimize()
                objective = self.sub_model.ObjVal
                time += self.sub_model.Runtime

            else:    
                self.sub_model.set_objective("max", self.sub_model.scal_prod(self.sub_vars, 
                    coeff))
            
                solution = self.sub_model.solve()
                objective = solution.get_objective_value()
                time += self.sub_model.solve_details.time

            if self.param.verbose: print(f"Magnanti & Wong: {objective}")

            self.data.sub_times.append(time) # Store time

        if self.param.verbose:
            print(f"Sub objective UB: {self.data.sub_objectives[-1]}")
            print("Sub solution time: {0:.4f}".format(time))



    def verification(self):

        """
            Verify the last optimal solution y by running the minimization problem of Eq 10
    
        :return solution: None or Float
        """

        constraints = self.verifier.add_constraints(
            self.verifier.scal_prod(self.z, self.A[i,:]) + 
            self.B[i,:]@self.last_minimum >= self.b[i] for i in range(self.B.shape[0]))
        self.verifier.set_objective("min", self.verifier.scal_prod(self.z, self.g))
        
        solution = self.verifier.solve()
        self.verifier.remove_constraints(constraints)

        return solution

    
    def generate_cut(self):
        """
            Generate the optimal cut with the solution (alpha, beta) from the Independent 
            Magnanti & Wong problem

        """
        # Set the value of alpha and beta
        if self.param.gurobi:
            varlist = self.sub_vars.tolist()
            alpha_k = np.array([var.X/self.param.weight_t for var in varlist[:len(self.B)]])
            beta_k = np.array([var.X/self.param.weight_t for var in varlist[len(self.B):]])
        else:
            alpha_k = np.array([var.solution_value/self.param.weight_t for var in self.sub_vars[:len(self.B)]])
            beta_k = np.array([var.solution_value/self.param.weight_t for var in self.sub_vars[len(self.B):]])

        # Create the cut as [-alpha_k@B,  ]
        cut = np.concatenate((- alpha_k @ self.B, 
            [alpha_k @ self.b + beta_k @ self.d]), axis=0)

        # If the maximum number of cuts (varphi) has been reached than remove the first cut from the set
        if len(self.set_of_cuts) >= self.param.max_cuts:
            self.set_of_cuts.pop(0)

        self.set_of_cuts.append(cut)
              

    def generate_quadratic(self, step: int) -> np.array:
        """
        Generate the QUBO formulation xTQx + qx + k

        :param step: int 
        :return (quadratic_term, linear_term, constant_term): (2-D np.array, 1-D np.array, float)
        """

        # Add variables to the problem for the new cut
        if step <= self.param.max_cuts:
            if self.param.real_var:
                self.num_qubits_a.append(1)
                self.master.continuous_var(ub=np.inf, name="x" + "_" + str(step))
            else:
                self.num_qubits_a.append(
                    # int(np.ceil(np.log2(((np.abs(self.set_of_cuts[-1][-1]) + sum(np.abs(self.set_of_cuts[-1][:-1])) + 
                    #     self.param.weight_p*np.power(2, self.param.num_qubits_p - 1))/self.param.weight_x) + 1)))
                    14
                    )
                self.master.binary_var_list(self.num_qubits_a[-1], name="x" + "_" + str(step))
        

        # Initialize the matrix for the constraints
        n_vars = self.master.number_of_variables if self.param.real_var else self.master.get_num_vars()
        quadratic_term = np.zeros((n_vars, n_vars))
        linear_term = np.concatenate((self.granularity_p/self.param.weight_t, np.zeros(self.n_binaries + 
            sum(self.num_qubits_a))), axis=0)
        constant_term = 0

        # Fill the quadratic vector with the set of cuts 
        for idx, cut in enumerate(self.set_of_cuts):

            constant_term += cut[-1]**2
            # Add artificial variables a
            if self.param.real_var:
                artificial_variables_vector = np.zeros(len(self.set_of_cuts))
                artificial_variables_vector[idx] = 1
            else:
                artificial_variables_vector = np.zeros(sum(self.num_qubits_a))     
                granularity_x = np.array([self.param.weight_x * np.power(2, i) for i in range(self.num_qubits_a[idx])])

                if idx == 0:
                    artificial_variables_vector[:self.num_qubits_a[idx]] = granularity_x
                else:
                    artificial_variables_vector[sum(self.num_qubits_a[:idx]):sum(self.num_qubits_a[:idx + 1])] = granularity_x

            quadratic_vector = np.concatenate((-self.granularity_p, cut[:-1], artificial_variables_vector), axis=0)

            linear_term += 2*cut[-1]*quadratic_vector

            # The Quadratic matrix is constructed by multiply together the hamiltoninan vectors of constraints
            quadratic_term += np.outer(quadratic_vector, quadratic_vector)

        if self.param.hamming:
            # Hamming distance from previous solution
            y_previous = self.master_variables[-1][self.granularity_p.size:self.granularity_p.size+self.n_binaries]

            hamming_vector = np.concatenate(
                (np.zeros(self.granularity_p.size), y_previous, np.zeros(artificial_variables_vector.size)), axis=0)
            
            constant_term += 0.5*sum(y_previous)
            linear_term -= hamming_vector
            quadratic_term += np.diag(0.5*hamming_vector)

        # Move to an upper triangular matrx
        for (i, j), q in np.ndenumerate(quadratic_term):
            if i < j:
                quadratic_term[i, j] = 2*q
                quadratic_term[j, i] = 0

        # Scaling everything by the maximum value
        # maximum = np.max([np.abs(quadratic_term).max(), np.abs(linear_term).max(), constant_term])
        # for (i, j), q in np.ndenumerate(quadratic_term):
        #     quadratic_term[i,j] /= maximum
        # linear_term /= maximum
        # constant_term /= maximum

        return quadratic_term, linear_term, constant_term

""" 
Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).

Contact: nicola.franco@iks.fraunhofer.de

"""

""" classic.py: Classical Robustness Analyzer Class."""

__copyright__   = "Copyright 2022, Fraunhofer IKS"


from docplex.mp.model import Model

class CompleteVerification():

    def __init__(self, matrices: list):

        """ 
        MILP optimization for comparison

        :param example: (int) if greater than zero than the verification run only on that specific example
        
        :return results: (list) solution and objective
        """
        self.A, self.B, self.C, self.b, self.d, self.g = matrices

        self.model = Model("Pure Classic") # Initialize the model
        # Define lower and upper boundaries
        self.z = self.model.continuous_var_list(self.C.shape[1], lb=-self.model.infinity, name="z") # Create continuous variables
        self.model.add_constraints_(self.model.scal_prod(self.z, self.C[i, :]) >= self.d[i] for i in range(self.C.shape[0]))
        # Create and add binaries variables to the problem
        if self.B.size > 0:
            self.y = self.model.binary_var_list(self.B.shape[1], name="y")
            self.model.add_constraints_(self.model.scal_prod(self.z, self.A[i,:]) + 
                self.model.scal_prod(self.y, self.B[i,:]) >= self.b[i] for i in range(self.B.shape[0]))  
        else:
            self.model.add_constraints_(self.model.scal_prod(self.z, self.A[i,:]) >= self.b[i] for i in range(self.A.shape[0]))
        
    def run(self, adversary: int):
        self.g[int(adversary - 10)] = - 1 

        self.model.minimize(self.g @ self.z)   # Set objective gT*x
        solution = self.model.solve()            # Optimize the model

        self.g[int(adversary - 10)] = 0

        if solution:
            return solution.get_objective_value(), self.model.solve_details.time
        else:            
            return float("nan"), float("nan")       


class IncompleteVerification():

    def __init__(self, matrices: list):

        """ 
        LP optimization for comparison

        :param example: (int) if greater than zero than the verification run only on that specific example
        
        :return results: (list) solution and objective
        """
        self.A, self.B, self.C, self.b, self.d, self.g = matrices 
        self.model = Model("Pure Classic") # Initialize the model
        # Define lower and upper boundaries
        self.z = self.model.continuous_var_list(self.C.shape[1], lb=-self.model.infinity, name="z") # Create continuous variables
        self.model.add_constraints_(self.model.scal_prod(self.z, self.C[i, :]) >= self.d[i] for i in range(self.C.shape[0])) 
        
        # Create and add binaries variables to the problem
        if self.B.size > 0:
            self.y = self.model.continuous_var_list(self.B.shape[1], lb=0, ub=1, name="y")
            self.model.add_constraints_(self.model.scal_prod(self.z, self.A[i,:]) + 
                self.model.scal_prod(self.y, self.B[i,:]) >= self.b[i] for i in range(self.B.shape[0]))
        else:
            self.model.add_constraints_(self.model.scal_prod(self.z, self.A[i,:]) >= self.b[i] for i in range(self.A.shape[0]))
        
    def run(self, adversary: int):
        self.g[int(adversary - 10)] = - 1 

        self.model.minimize(self.g @ self.z)   # Set objective gT*x
        solution = self.model.solve()            # Optimize the model

        self.g[int(adversary - 10)] = 0

        if solution:
            return solution.get_objective_value(), self.model.solve_details.time
        else:
            return float("nan"), float("nan")
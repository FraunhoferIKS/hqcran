""" 
Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).

Contact: nicola.franco@iks.fraunhofer.de

"""

import os
import numpy as np

import time
# Quantum MILP
from decomposition import dantzigwolfe, benders
from classic import CompleteVerification
from constraints import ConstraintsGenerator
from parameters import Param, Data

class Test:
    def __init__(self):

        self.sample: int = 0
        self.label: int = 0
        self.robust: bool = True
        self.incomplete: bool = False
        self.time: float = 0
        self.data: list = []



class QMILP:
    def __init__(self, parameters: Param):
        
        """
        Defines the penality weights and the number of qubits to approximate the variables 

        :param dataset: input dataset
        :param model_path: (str) neural network model path
        :param parameters: (Param) 
        """

        self.parameters = parameters # Store parameters
        
        self.model_name = os.path.splitext(os.path.basename(self.parameters.onnxpath))[0]
        dataset_path = os.path.abspath(os.path.join(os.path.abspath(''), self.parameters.dataset))
        self.dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.dataset = np.genfromtxt(dataset_path, delimiter=',') # Load the dataset
        self.processes = 9
        self.results = [] # Results


    def solve(self):
        """
        Main loop

        """
        
        start = self.parameters.start if self.parameters.start else 0
        end = self.parameters.end if self.parameters.end else len(self.dataset)
        assert start < end, "Start example should be lower than the ending one"
        
        n_robust, total_time, n_of_samples = 0, 0, end - start
        for idx in range(start, end):
            
            test = Test()
            test.sample = idx
            image = self.dataset[idx][1:]
            test.label = int(self.dataset[idx][0])

            timer = time.perf_counter()
            constraints = ConstraintsGenerator(image, test.label, self.parameters) # Generate the constraints
            test.time = time.perf_counter() - timer
            
            print('[Sample {}]: label: {}, predicted: {}'.format(idx, test.label, constraints.prediction))

            if test.label != constraints.prediction:
                print('Skip the wrongly classified sample')
                continue

            if self.parameters.crown_ibp and constraints.global_lb.min() >= 0 and 'cplex' in self.parameters.solver:
                print(f"verified with init bound! time: {test.time}")
                test.incomplete = True
            else:
                
                matrices = constraints.generate_constraints()
                bounds = constraints.get_bounds()
                print(f'{matrices[1].shape[1]} unstable neurons')

                # Hybrid Optimization initialization
                if self.parameters.decomposition == 'dantzig-wolfe':
                    hybrid_problem = dantzigwolfe.DWDecomposition(
                        self.parameters, matrices, bounds
                        )
                elif self.parameters.decomposition == 'benders':
                    self.parameters.compute_qubits_p(constraints.max_bounds_value)
                    hybrid_problem = benders.BendersDecomposition(
                        self.parameters, matrices
                        )
                else:
                    raise ValueError('Decomposition method not supported')
                
                if self.parameters.classic or matrices[1].size == 0: # Pure Classic optimization initialization
                    complete_problem = CompleteVerification(matrices)

                # Iterate over all possible adversarial classes
                adversarial_examples = list(range(10)) # Generate adversarial classes
                adversarial_examples.remove(test.label)
                for adversary in adversarial_examples:
                    if matrices[1].size:
                        data = hybrid_problem.run(adversary=adversary)
                    else:
                        complete_objective, complete_time = complete_problem.run(adversary=adversary)
                        data = hybrid_problem.data
                        data.solution = complete_objective
                        data.master_objectives = [complete_objective]
                        data.sub_objectives = [complete_objective]
                        data.total_time = complete_time
                        data.master_times = [complete_time/2]
                        data.sub_times = [complete_time/2]

                    print('[Sample {}]: test against label {}, hybrid solution: {:0.3f}, solve time: {:0.3f} sec., iterations: {}, qubits: {}'
                        .format(idx, adversary, data.solution, data.total_time, len(data.sub_objectives), data.num_qubits[-1]))

                    if self.parameters.classic: # Pure Classic Optimization
                        complete_objective, complete_time = complete_problem.run(adversary) 
                        print('[Sample {}]: test against label {}, exact solution: {:0.3f}, solve time: {:0.3f} sec.'.format(
                            idx, 
                            adversary, 
                            complete_objective, 
                            complete_time))
                        data.complete_objective = complete_objective
                        data.complete_time = complete_time

                    if data.solution < 0:
                        test.robust = False
                    test.time += data.total_time
                    test.data.append(data.__dict__)
                    if data.solution < 0 and self.parameters.crown_ibp:
                        break
            
            if test.robust: 
                n_robust += 1
            total_time += test.time
            self.results.append(test.__dict__)

        print('\n\n# of robust samples: {}/{}, average time: {}'.format(
            n_robust, n_of_samples, total_time/n_of_samples))
        
        return self.results


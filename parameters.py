""" 
Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).

Contact: nicola.franco@iks.fraunhofer.de

"""

import os
import numpy as np
# Qiskit
from qiskit_optimization.algorithms import CplexOptimizer, MinimumEigenOptimizer
from qiskit.algorithms.optimizers import SPSA
from qiskit.providers.aer import AerError
from qiskit.algorithms import QAOA
from qiskit_optimization.runtime import QAOAClient
from qiskit import Aer, IBMQ
from docplex.mp.context import Context
# D-Wave
from dwave.system import DWaveCliqueSampler, LeapHybridSampler, CutOffComposite
import neal

class Param:
    def __init__(self):

        self.netname: str = None                # Set the network path
        self.onnxpath: str = None               
        self.torchpath: str = None
        self.dataset: str = None                # Set the dataset path
        self.output: str = None                 # Set the output folder
        self.solver: str = None                 # Set the solver to: cplex, ibm_simulator, ibm_cloud, dwave_hybrid, dwave_qpu, dwave_simulator
        self.epsilon: float = 0.001             # Set epsilon
        self.weight_t: int = 1                  # Set the penalization value for the quadratic term
        self.weight_x: float = 0.01             # Set the weights to properly adjust the approximation of x
        self.weight_p: float = 0.01             # Set the weights to properly adjust the approximation of p
        self.num_qubits_p: int = 30             # Define the number of qubits to approximate the real variable p
        self.num_qubits_x: int = 30             # Set the number of quibits to approximate the constraints added to the master problem
        self.sub_bound: int = 10                # Set the maximum value for the sub bounds 
        self.objectives_gap: float = 0.1        # Gap between the master and sub objective
        self.start: int = None                  # Set the starting test
        self.end: int = None                    # Set the ending test
        self.steps: int = 100                   # Set the maximum number of steps for the hybrid algorithm
        self.classic: bool = True               # Set the pure classic solve to True if you want to have a comparison
        self.dwave_reads: int = 100             # Set the number of reads of the D-Wave QPU system
        self.threads: int = 32                  # Set the maximum number of Threads
        self.verbose: bool = False              # Set debug to True if you want to visualize every output
        self.real_var: bool = False             # Set the real variables for debugging the benders decomposition
        self.qubo: bool = True                  # Set the QUBO formulation to True (False works only with real variables)
        self.hamming: bool = False              # Set the strenght factor for the Hamming distance from the previous solution
        self.magnanti: bool = True              # Set the generation of Magnanti & Wong cuts
        self.max_cuts: int = np.inf             # 
        self.quantum_solver = None
        self.gurobi: bool = False
        self.crown_ibp: bool = True                # Set the CROWN-IBP bounderies to True    
        self.unbalanced_penalities:list = 3*[1]
        self.decomposition: str = 'benders'
   

    def solver_initialization(self):        
        # Initialize the quantum solver
        if self.solver == 'cplex':
            context = Context.make_default_context()
            context.cplex_parameters.threads = self.threads
            self.quantum_solver = CplexOptimizer(cplex_parameters=context.cplex_parameters)
        
        elif self.solver == 'ibm_simulator':

            """ Use your hardware """
            backend = Aer.get_backend('aer_simulator')
            # try:
            #     backend.set_options(device='GPU')
            #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            # except AerError as e:
            #     print(e)
            
            # spsa = SPSA(maxiter=250)
            # qaoa = QAOA(optimizer=spsa, reps=2, quantum_instance=backend)
            self.quantum_solver = backend #MinimumEigenOptimizer(qaoa)

        elif self.solver == 'ibm_cloud':
            
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
            
            self.quantum_solver = provider
            
        elif self.solver == 'dwave_hybrid': 
            self.quantum_solver = LeapHybridSampler(solver={'category': 'hybrid'})
        
        elif self.solver == 'dwave_qpu':
            self.quantum_solver = CutOffComposite(DWaveCliqueSampler(solver={'topology__type': 'pegasus'}), 0.05) #Clique
        
        elif self.solver == 'dwave_simulator':
            self.quantum_solver = neal.SimulatedAnnealingSampler()
        else:
            raise NameError('Wrong Solver Type: classic, ibm_simulator, ibm_cloud, dwave_hybrid, dwave_qpu, dwave_simulator')

    def compute_qubits_p(self, max_bounds_value: float):
        try:
            self.num_qubits_p = int(
                1 + np.ceil(np.log2(1 + 2*max_bounds_value/self.weight_p))
                )
        except:
            self.num_qubits_p = int(1 +  np.ceil(np.log2(200/self.weight_p)))
            
        print(f'{self.num_qubits_p} qubits to approximate the objective')

class Data:
    def __init__(self):
        self.adversary: int = 0
        self.solution: float = float('nan')
        self.gap: float = float('nan')
        self.master_objectives: list = []
        self.sub_objectives: list = []
        self.total_time: float = 0
        self.master_times: list = []
        self.sub_times: list = []
        self.num_qubits: list = [0]

class DWData:
    def __init__(self):
        self.adversary: int = 0
        self.solution: float = float('nan')
        self.gap: float = float('nan')
        self.master_objectives: list = []
        self.sub_objectives: list = []
        self.dual_master_obj: list = []
        self.sub_real_obj:list = []
        self.sub_bin_obj: list = []
        self.total_time: float = 0
        self.master_times: list = []
        self.sub_times: list = []
        self.real_times:list = []
        self.bin_times:list = []
        self.num_qubits:list = [0]
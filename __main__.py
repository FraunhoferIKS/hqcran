""" 
Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).

Contact: nicola.franco@iks.fraunhofer.de

"""

"""__main__.py: HQ-CRAN Hybrid Quantum-Classical Robustness Analyzer for Neural Networks """

__copyright__   = "Copyright 2022, Fraunhofer IKS"

import optimization, parameters
from pathlib import Path
from pprint import pprint
import yaml
import argparse
import os
import numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def isnetworkfile(fname):
    _, ext = os.path.splitext(fname)
    if ext not in ['.onnx']: #TODO: '.pyt', '.meta', '.tf', '.pb' 
        raise argparse.ArgumentTypeError('only .onnx format supported')
    return fname

def validsolver(solver):
    if solver not in ['cplex', 'ibm_simulator', 'ibm_cloud', 'dwave_hybrid', 'dwave_qpu', 'dwave_simulator' ]:
        raise argparse.ArgumentTypeError('the solver can only be: cplex, ibm_simulator, ibm_cloud, dwave_hybrid, dwave_qpu, dwave_simulator')
    return solver

config = parameters.Param()

parser = argparse.ArgumentParser(description='HQ-CRAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, help='path to YAML parameters file.')
parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .onnx')
parser.add_argument('--dataset', type=str, default=config.dataset, help='specify the dataset path')
parser.add_argument('--output', type=str, default=config.output, help='specify the output folder')
parser.add_argument('--epsilon', type=float, default=config.epsilon, help='the epsilon for L_infinity perturbation')
parser.add_argument('--decomposition', type=str, default=config.decomposition, help='the decomposition method can be: dantzigwolfe, benders')
parser.add_argument('--solver', type=validsolver, default=config.solver, help='the solver can only be: cplex, ibm_simulator, ibm_cloud, dwave_hybrid, dwave_qpu, dwave_simulator')
parser.add_argument('--weigth_p', type=float, default=config.weight_p, help='the granularity weight to approximate the variable p')
parser.add_argument('--weight_x', type=float, default=config.weight_x, help='the granularity weight to approximate the variable x')
parser.add_argument('--num_qubits_p', type=float, default=config.num_qubits_p, help='the number of qubits to approximate p')
parser.add_argument('--num_qubits_x', type=float, default=config.num_qubits_x, help='the number of qubits to approximate x')
parser.add_argument('--sub_bound', type=float, default=config.sub_bound, help='the bounds of the dual algorithm')
parser.add_argument('--steps', type=float, default=config.steps, help='the number of steps of the hybrid approach')
parser.add_argument('--dwave_reads', type=float, default=config.dwave_reads, help='the number of reads for the D-Wave QPU system')
parser.add_argument('--start', type=int, default=config.start, help='run the verification from one specific example')
parser.add_argument('--end', type=int, default=config.end, help='run the verification until one specific example')
parser.add_argument('--threads', type=int, default=config.threads, help='the number of maximum threads')
parser.add_argument('--classic', type=str2bool, default=config.classic, help='the pure classical solver')
parser.add_argument('--verbose', type=str2bool, default=config.verbose, help='display the information')

args = parser.parse_args()
if args.config:
    with open(args.config, 'r') as config_file:
        loaded_args = yaml.safe_load(config_file)
        for k, v in loaded_args.items():
            setattr(config, k, v)
else:
    for k, v in vars(args).items():
        setattr(config, k, v)

assert config.solver in ['cplex', 'ibm_simulator', 'vqe', 'ibm_cloud', 'dwave_hybrid', 'dwave_qpu', 'dwave_simulator' ]
config.solver_initialization()

assert config.onnxpath, 'a network has to be provided for analysis.'
assert 'mnist' in config.dataset, 'MNIST is the only supported dataset'
config.output = Path(config.output).resolve()
assert config.output, 'a valid output path for the results should be provided'
if config.real_var and config.solver in ['ibm_simulator', 'ibm_cloud', 'dwave_hybrid', 'dwave_qpu', 'dwave_simulator' ]:
    raise Exception('real variables only with cplex')
if not config.qubo and config.solver in ['ibm_simulator', 'ibm_cloud', 'dwave_hybrid', 'dwave_qpu', 'dwave_simulator' ]:
    raise Exception('QUBO formulation required with {}'.format(config.solver))
if not config.real_var and not config.qubo:
    raise Exception('QUBO formulation required without real variables')
import time
print("Test configuration: ")
pprint(vars(config))
start = time.time()

## Main algorithm
problem = optimization.QMILP(parameters=config)
results = problem.solve() 
print("Execution Time: ", time.time() - start)

results_name = 'mnist_{}_{}_{}_{}_{}_epsilon_{}_bound_{}_gap_{}'.format(
    config.decomposition, config.start, config.end, config.solver,
    config.netname, config.epsilon, config.sub_bound,
    config.objectives_gap
)

if config.real_var:
    results_name = 'real_' + results_name + '_classic_{}.npy'.format(config.classic)
else:
    results_name += '_x_{:.2f}_p_{:.2f}_classic_{}.npy'.format(
        config.weight_x, config.weight_p, config.classic
    )

np.save(config.output/Path(results_name), results)
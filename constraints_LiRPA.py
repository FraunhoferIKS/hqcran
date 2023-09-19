""" 
Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).

Contact: nicola.franco@iks.fraunhofer.de

"""

import copy

from collections import defaultdict, OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundRelu
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import reduction_sum, stop_criterion_sum, stop_criterion_min

# from constraints import ConstraintsGenerator

class LiRPAConvNet:
    def __init__(self, torch_netpath, model_name, device='cuda', simplify=False, in_size=(1, 3, 32, 32),
                 conv_mode='patches', deterministic=False, c=None):
        """
        convert pytorch model to auto_LiRPA module
        """
        model_ori = load_model(torch_netpath, model_name)

        net = copy.deepcopy(model_ori)
        layers = list(net.children())
        self.simplify = False
        self.c = c
        self.layers = layers
        self.input_shape = in_size
        self.net = BoundedModule(net, torch.zeros(in_size, device=device), 
            bound_opts={'relu': 'adaptive', 'deterministic': deterministic, 'conv_mode': conv_mode}, device=device)
        self.net.eval()
        self.needed_A_dict = None
        self.pool = None   # For multi-process.
        self.pool_result = None
        self.pool_termination_flag = None

    def build_the_model(self, input_domain, x, stop_criterion_func=stop_criterion_sum(0)):

        lr_init_alpha = 0.1
        init_iteration = 100
        share_slopes = False
        optimizer = "adam"
        lr_decay = 0.98
        loss_reduction_func = "sum"
        get_upper_bound = False
        
        self.x = x
        self.input_domain = input_domain

        slope_opt = None

        loss_reduction_func = self.reduction_str2func(loss_reduction_func)

        # first get CROWN bounds
        # Reference bounds are intermediate layer bounds from initial CROWN bounds.
        lb, ub, aux_reference_bounds = self.net.init_slope((self.x,), share_slopes=share_slopes, c=self.c, bound_upper=False)
        # print('initial CROWN bounds:', lb, ub)
        if stop_criterion_func(lb).all().item():
            # Fast path. Initial CROWN bound can verify the network.
            if not self.simplify:
                return None, lb[-1], None, None, None, None, None, None, None, None, None, None
            else:
                return None, lb[-1].item(), None, None, None, None, None, None, None, None, None, None
        self.net.set_bound_opts({'optimize_bound_args': {'ob_iteration': init_iteration, 'ob_beta': False, 'ob_alpha': True,
                                'ob_alpha_share_slopes': share_slopes, 'ob_optimizer': optimizer,
                                'ob_early_stop': False, 'ob_verbose': 0,
                                'ob_keep_best': True, 'ob_update_by_layer': True,
                                'ob_lr': lr_init_alpha, 'ob_init': False,
                                'ob_loss_reduction_func': loss_reduction_func,
                                'ob_stop_criterion_func': stop_criterion_func,
                                'ob_lr_decay': lr_decay}})
        lb, ub = self.net.compute_bounds(x=(x,), IBP=False, C=self.c, method='CROWN-Optimized', return_A=False,
                                                bound_upper=False, aux_reference_bounds=aux_reference_bounds)

        slope_opt = self.get_slope(self.net)[0]  # initial with one node only
        # build a complete A_dict
        # self.layer_names = list(A_dict[list(A_dict.keys())[-1]].keys())[2:]
        # self.layer_names.sort()

        # update bounds
        # print('initial alpha-CROWN bounds:', lb, ub)
        primals, duals, mini_inp = None, None, None
        # mini_inp, primals = self.get_primals(self.A_dict)
        lb, ub, pre_relu_indices = self.get_candidate(self.net, lb, lb + 99)  # primals are better upper bounds
        mask, lA = self.get_mask_lA_parallel(self.net)

        if not self.simplify or stop_criterion_func(lb[-1]):
            history = [[[], []] for _ in range(len(self.net.relus))]
            return ub[-1], lb[-1], mini_inp, duals, primals, mask[0], lA[0], lb, ub, pre_relu_indices, slope_opt, history

        # for each pre-relu layer, we initial 2 lists for active and inactive split
        history = [[[], []] for _ in range(len(self.net.relus))]

        if get_upper_bound:
            self.needed_A_dict = defaultdict(set)
            self.needed_A_dict[self.net.output_name[0]].add(self.net.input_name[0])
        
        return ub[-1].item(), lb[-1].item(), mini_inp, duals, primals, mask[0], lA[0], lb, ub, pre_relu_indices, slope_opt, history

    def reduction_str2func(self, reduction_func):
        if type(reduction_func) == str:
            if reduction_func == 'min':
                return reduction_min
            elif reduction_func == 'max':
                return reduction_max
            elif reduction_func == 'sum':
                return reduction_sum
            elif reduction_func == 'mean':
                return reduction_mean
            else:
                raise NotImplementedError(f'Unknown reduction_func {reduction_func}')
        else:
            return reduction_func

    def get_slope(self, model):
        if len(model.relus) == 0:
            return [None]

        # slope has size (2, spec, batch, *shape). When we save it, we make batch dimension the first.
        # spec is some intermediate layer neurons, or output spec size.
        batch_size = next(iter(model.relus[0].alpha.values())).size(2)
        ret = [defaultdict(dict) for i in range(batch_size)]
        for m in model.relus:
            for spec_name, alpha in m.alpha.items():
                # print(f'save layer {m.name} start_node {spec_name} shape {alpha.size()} norm {alpha.abs().sum()}')
                for i in range(batch_size):
                    # each slope size is (2, spec, 1, *shape).
                    ret[i][m.name][spec_name] = alpha[:,:,i:i+1,:]
        return ret

    def get_candidate(self, model, lb, ub):
        # get the intermediate bounds in the current model and build self.name_dict which contains the important index
        # and model name pairs

        lower_bounds = []
        upper_bounds = []
        self.pre_relu_indices = []
        i = 0
        # build a name_dict to map layer idx in self.layers to BoundedModule
        self.name_dict = {}

        for layer in model.relus:
            lower_bounds.append(layer.inputs[0].lower.detach())
            upper_bounds.append(layer.inputs[0].upper.detach())
            self.name_dict[i] = layer.inputs[0].name
            self.pre_relu_indices.append(i)
            i += 1

        # Also add the bounds on the final thing
        lower_bounds.append(lb.view(1, -1).detach())
        upper_bounds.append(ub.view(1, -1).detach())

        return lower_bounds, upper_bounds, self.pre_relu_indices
    
    def get_mask_lA_parallel(self, model):
        if len(model.relus) == 0:
            return [None], [None]
        # get the mask of status of ReLU, 0 means inactive neurons, -1 means unstable neurons, 1 means active neurons (this is not used).
        # get lower A matrix of ReLU
        mask, lA = [], []
        for this_relu in model.relus:
            # 1 is unstable neuron, 0 is stable neuron.
            mask_tmp = torch.logical_and(this_relu.inputs[0].lower < 0, this_relu.inputs[0].upper > 0).float()
            mask.append(mask_tmp.reshape(mask_tmp.size(0), -1))
            if this_relu.lA is not None:
                lA.append(this_relu.lA.squeeze(0))
            else:
                # It might be skipped due to inactive neurons.
                lA.append(None)

        ret_mask, ret_lA = [], []
        for i in range(mask[0].size(0)):
            ret_mask.append([j[i:i+1] for j in mask])
            ret_lA.append([j[i:i+1] if j is not None else None for j in lA])
        return ret_mask, ret_lA


class ReLU2x20(nn.Module):
    def __init__(self, nclasses: int = 10, input_size: int = 784):
        super(ReLU2x20, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, nclasses)

    def forward(self, x):
        ''' Layer 1 '''
        x = x.view(-1, self.input_size)    
        x = F.relu(self.fc1(x))
        ''' Layer 2 '''        
        x = F.relu(self.fc2(x))

        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

def mnist_2_20():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
    )
    return model

def mnist_3_30():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 30),
        nn.ReLU(),
        nn.Linear(30, 30),
        nn.ReLU(),
        nn.Linear(30, 10),
        nn.ReLU(),
    )
    return model

def mnist_6_100():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100, 10),
        # nn.ReLU(),
        # nn.Linear(10,10, bias=False)
    )
    return model

def load_model(path, model_name, weights_loaded=True):
    """
    Load the model architectures and weights
    """
    # You can customize this function to load your own model based on model name.
    model_ori = eval(model_name)()
    # print(model_ori)

    if not weights_loaded:
        return model_ori

    sd = torch.load(path)#, map_location=torch.device('cpu'))
    if 'state_dict' in sd:
        sd = sd['state_dict']
    if type(sd) == list:
        sd = sd[0]
    elif type(sd) == OrderedDict:
        pass
    else:
        raise NotImplementedError
    model_ori.load_state_dict(sd)

    return model_ori

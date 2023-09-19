""" 
Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).

Contact: nicola.franco@iks.fraunhofer.de

"""

import onnx_translator
from parameters import Param

from constraints_LiRPA import LiRPAConvNet
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import reduction_sum, stop_criterion_sum, stop_criterion_min

import numpy as np
import os
import torch
import onnxruntime
import onnx

class ConstraintsGenerator:
    """
    Generate the constraints matrices of the MILP problem (B, A, C) and the corresponding vectors (g, b, d)

    minimize   g^T@z
    s.t.    B@y + A@z >= b
            C@z >= d

    """
    def __init__(self, image, label, parameters: Param):
        """
        Arguments
        --------
        :param image: numpy.ndarray
        :param label: int
        :param parameters: Param object
        """

        # Load ONNX model
        self.param = parameters
        self.max_bounds_value = 0
        self.netname = parameters.netname
        model_path = os.path.abspath(os.path.join(os.path.abspath(''), parameters.onnxpath)) # Set the model path
        self.torch_path = parameters.torchpath
        self.model = onnx.load(model_path)
        self.session = onnxruntime.InferenceSession(model_path, None)
        
        # We use the ERAN library to load and convert a onnx model into a IR 
        # For more information: https://github.com/eth-sri/eran
        translator = onnx_translator.ONNXTranslator(self.model, False)
        self.operations, self.resources = translator.translate()
        self.ibp_boundaries = []
        # Load image
        self.image = np.array(image/255.0, dtype=np.float64)
        self.label = label
        self.input_region = self.generate_input_boundaries(parameters.epsilon)

        # Predicts the label
        self.prediction = self.predict(image)
        self.num_qubits_p = 0
        _, self.global_lb, _, _, _, _, _, self.lower_bounds, self.upper_bounds, _, _, _ = self.compute_bounds(parameters.epsilon)

    def generate_input_boundaries(self, epsilon):
        """
        Generate input boundaries

        Arguments
        ------------
        :param epsilon: float
            epsilon value
        
        Returns
        -------
        :return: numpy.ndarray
            input boundaries
        """
        perturbation = epsilon
        upper_image = np.array(self.image + perturbation, dtype=np.float64)
        lower_image = np.array(self.image - perturbation, dtype=np.float64)
        # L infinity norm
        input_region = np.column_stack([lower_image, upper_image]) 
        # Bounds the region between 0-255
        input_region[:, 0] = [0 if l_bound < 0 else l_bound for l_bound in input_region[:, 0]]
        input_region[:, 1] = [1 if u_bound > 1 else u_bound for u_bound in input_region[:, 1]]

        return input_region
    
    def compute_bounds(self, eps):
        """ 
        Compute the bounds of the network
        
        Arguments
        ------------
        :param eps: float
            epsilon value
        
        Returns
        -------
        :return: 
        """
        data = torch.tensor(self.image.reshape(1, 28, 28), dtype=torch.float32)
        num_outputs = 10
        labels = torch.tensor([self.prediction]).long()
        c = torch.eye(num_outputs).type_as(data)[labels].unsqueeze(1) - torch.eye(num_outputs).type_as(data).unsqueeze(0)
        I = (~(labels.data.unsqueeze(1) == torch.arange(num_outputs).type_as(labels.data).unsqueeze(0)))
        # Remove spec to self.
        c = (c[I].view(data.size(0), num_outputs - 1, num_outputs))

        data_ub = data + eps
        data_lb = data - eps
        domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)

        model = LiRPAConvNet(self.torch_path, self.netname, device='cpu', simplify=False, in_size=data.shape,
                conv_mode='patches', deterministic=True, c=c)

        ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=data_lb, x_U=data_ub)
        x = BoundedTensor(data, ptb)
       
        return model.build_the_model(domain, x, stop_criterion_func=stop_criterion_min(0))
    
    def get_bounds(self):
        """ 
        Get the bounds of the network
        
        Returns
        -------
        :return: tuple
            (lower_bounds, upper_bounds)
        """
        lower_bounds = np.concatenate([bound[:, 0] for bound in self.ibp_boundaries])
        upper_bounds = np.concatenate([bound[:, 1] for bound in self.ibp_boundaries])

        return (lower_bounds, upper_bounds)

    def predict(self, image):
        """ 
        Predict the label of the image
        
        Arguments
        ------------
        :param image: input image
        
        :return: int
            predicted label
        """

        image = image.reshape(1, 1, 28, 28)/255.0
        input_name = self.session.get_inputs()[0].name
        raw_result = self.session.run([], {input_name: np.array(image, dtype=np.float32)})
        _, predicted = torch.max(torch.as_tensor(raw_result[0]), 1)
        del self.session

        return predicted.item()

    def generate_constraints(self):
        
        """ 
        Generate the constraints matrices 

        minimize   g^T@z
        s.t.    B@y + A@z >= b
                C@z >= d

        Arguments
        ------------
        :param image: input image
        :param neural_network_weights: neural network weights


        :return: list of array [B, A, C, b, d, g] 
        """
        last_layer = len(self.operations) - 1
        # A, B & b initialization
        A_top = np.array([], dtype=np.float64)
        A_bottom = np.array([], dtype=np.float64)
        B_top = np.array([], dtype=np.float64)
        B_bottom = np.array([], dtype=np.float64)
        b_top = np.array([], dtype=np.float64)
        b_bottom = np.array([], dtype=np.float64)

        bound_idx = 0

        for idx, op in enumerate(self.operations):

            # The first rows of C@z >= d
            if 'Placeholder' in op:
                input_size = self.resources[idx]['deepzono'][2]
                boundaries = self.input_region
                self.ibp_boundaries.append(boundaries)
                # C matrix initialization
                
                C_top = np.concatenate((np.identity(boundaries.shape[0]), - np.identity(boundaries.shape[0])))
                C_top.astype( dtype=np.float64)
                # d vector initialization
                d_top = np.concatenate((boundaries[:,0], -boundaries[:, 1]), dtype=np.float64)
            
            # The rows of C@z >= d
            elif 'Gemm' in op:

                weights = np.array(self.resources[idx]['deepzono'][0], dtype=np.float64)
                biases = np.array(self.resources[idx]['deepzono'][1], dtype=np.float64)

                if self.param.crown_ibp:
                    l_bounds, u_bounds = self.lower_bounds[bound_idx].numpy()[0], self.upper_bounds[bound_idx].numpy()[0]
                else:
                    # Intervals arithmetics
                    weights_negative = np.minimum(weights, 0)
                    biases_negative = np.minimum(biases, 0)
                    weights_positive = np.maximum(weights, 0)
                    biases_positive = np.maximum(biases, 0)

                    l_bounds = weights_positive@boundaries[:, 0] + weights_negative@boundaries[:, 1] + biases_negative
                    u_bounds = weights_positive@boundaries[:, 1] + weights_negative@boundaries[:, 0] + biases_positive
                
                boundaries = np.column_stack((l_bounds, u_bounds))
                
                bound_idx += 1
                # C matrix building
                C_top = np.column_stack((C_top, np.zeros((C_top.shape[0], weights.shape[0]))))

                if idx == 1:
                    # C matrix building
                    C_top = np.vstack((C_top, np.column_stack((-weights, np.identity(weights.shape[0])))))
                    C_bottom = np.column_stack((np.zeros(weights.shape), np.identity(weights.shape[0])))
                    # d initialization
                    d_top = np.concatenate((d_top, biases), dtype=np.float64)
                    d_bottom = np.zeros(weights.shape[0], dtype=np.float64)
                else:
                    ## C matrix building
                    # C top
                    zeros_from_left_C_top = np.zeros((weights.shape[0], C_top.shape[1] - weights.shape[1] - weights.shape[0]))                   
                    new_row_C_top = np.column_stack((zeros_from_left_C_top, - weights, np.identity(weights.shape[0])))
                    C_top = np.vstack((C_top, new_row_C_top))
                    # C bottom
                    C_bottom = np.column_stack((C_bottom, np.zeros((C_bottom.shape[0], weights.shape[0]))))
                    zeros_from_left_C_bottom = np.zeros((weights.shape[0], C_bottom.shape[1] - weights.shape[0]))                    
                    new_row_C_bottom = np.column_stack((zeros_from_left_C_bottom, np.identity(weights.shape[0])))
                    C_bottom = np.vstack((C_bottom, new_row_C_bottom))
                    # d bottom building
                    d_top = np.concatenate((d_top, biases))
                    d_bottom = np.concatenate((d_bottom, np.zeros(weights.shape[0])), dtype=np.float64)


            # The rows of B@y + A@z >= b
            elif 'Relu' in op: #and idx < last_layer:

                # Add zeros matrix to the right hand-side of A
                if idx > 2: 
                    A_top = np.column_stack((A_top, np.zeros((A_top.shape[0], weights.shape[0]))))
                else:
                    A_top = np.zeros((0, weights.shape[1] + weights.shape[0]), dtype=np.float64)

                for j, bounds in enumerate(boundaries):

                    # Stable active units
                    if bounds[0] > 0:

                        ## A matrix building
                        position_vector = np.zeros((1, weights.shape[0]), dtype=np.float64)
                        position_vector[0, j] = - 1

                        if idx > 2:
                            zeros_from_left_A_top = np.zeros((1, A_top.shape[1] - weights.shape[0] - weights.shape[1]), dtype=np.float64)
                            new_row_A_top = np.column_stack((zeros_from_left_A_top, np.expand_dims(weights[j], axis=0), position_vector))
                        else:
                            new_row_A_top = np.column_stack((np.expand_dims(weights[j], axis=0), position_vector))
                        A_top = np.vstack((A_top, new_row_A_top))

                        if B_top.size:
                            B_top = np.vstack((B_top, np.zeros(B_top.shape[1])))
                        elif B_top.shape[0]:
                            B_top = np.zeros((B_top.shape[0] + 1, 0))
                        else:
                            B_top = np.zeros((1, 0))
                        
                        if B_bottom.size:
                            B_bottom = np.vstack((B_bottom, np.zeros(B_bottom.shape[1])))
                        elif B_bottom.shape[0]:
                            B_bottom = np.zeros((B_bottom.shape[0] + 1, 0))
                        else:
                            B_bottom = np.zeros((1, 0))
                        
                        b_top = np.concatenate((b_top, [- biases[j]]), dtype=np.float64)
                        b_bottom = np.concatenate((b_bottom, [- bounds[1]]), dtype=np.float64)

                    # Stable inactive units
                    elif bounds[1] <= 0:
                        # turn off the stable inactive boundaries
                        boundaries[j] = np.zeros(2)
                        if B_bottom.size:
                            B_bottom = np.vstack((B_bottom, np.zeros(B_bottom.shape[1])))
                        elif B_bottom.shape[0]:
                            B_bottom = np.zeros((B_bottom.shape[0] + 1, 0))
                        else:
                            B_bottom = np.zeros((1, 0))

                        b_bottom = np.concatenate((b_bottom, [0]), dtype=np.float64)

                    # Unstable units
                    else:
                        ## A matrix building
                        position_vector = np.zeros((1, weights.shape[0]), dtype=np.float64)
                        position_vector[0, j] = - 1
                        
                        if idx > 2:
                            zeros_from_left_A_top = np.zeros((1, A_top.shape[1] - weights.shape[0] - weights.shape[1]), dtype=np.float64)
                            new_row_A_top = np.column_stack((zeros_from_left_A_top, np.expand_dims(weights[j], axis=0), position_vector))
                        else:
                            new_row_A_top = np.column_stack((np.expand_dims(weights[j], axis=0), position_vector))

                        A_top = np.vstack((A_top, new_row_A_top))
                        
                        if B_top.shape[0]:
                            B_top = np.column_stack((B_top, np.zeros((B_top.shape[0], 1))))
                            B_top = np.vstack((B_top, np.column_stack((np.zeros((1, B_top.shape[1] - 1)), [bounds[0]]))))
                        else:
                            B_top = np.array([bounds[0]], dtype=np.float64, ndmin=2)
                        
                        if B_bottom.shape[0]:
                            B_bottom = np.column_stack((B_bottom, np.zeros((B_bottom.shape[0], 1))))
                            B_bottom = np.vstack((B_bottom, np.column_stack((np.zeros((1, B_bottom.shape[1] - 1)), [bounds[1]]))))
                        else:
                            B_bottom = np.array([bounds[1]], dtype=np.float64, ndmin=2)

                        b_top = np.concatenate((b_top, [bounds[0] - biases[j]]), dtype=np.float64)
                        b_bottom = np.concatenate((b_bottom, [0]), dtype=np.float64)
            
                # else:
                ## A matrix building
                if A_bottom.size:
                    # Create zeros matrix to insert from left
                    zeros_from_left_A_bottom = np.zeros((weights.shape[0], A_bottom.shape[1]), dtype=np.float64)
                    # Add zeros matrix to the right hand-side of A
                    A_bottom = np.column_stack((A_bottom, np.zeros((A_bottom.shape[0], weights.shape[0]))))
                    # Insert new rows
                    new_row_A_bottom = np.column_stack((zeros_from_left_A_bottom, - np.identity(weights.shape[0])))
                    A_bottom = np.vstack((A_bottom, new_row_A_bottom))
                else:
                    # A matrix initialization
                    A_bottom = np.column_stack((np.zeros(weights.shape), - np.identity(weights.shape[0])))

                self.ibp_boundaries.append(boundaries)
        
        # Compose the first constraint
        if B_top.size and B_bottom.size:
            B = np.vstack((B_top, B_bottom))
        elif B_top.size:
            B = B_top
        else:
            B = B_bottom

        if A_top.size:
            A = np.vstack((A_top, A_bottom))
        else:
            A = A_bottom

        b = np.concatenate((b_top, b_bottom), dtype=np.float64)

        C = np.vstack((C_top, C_bottom)) # Compose the second constraint
        d = np.concatenate((d_top, d_bottom), dtype=np.float64)

        if A.shape[1] < C.shape[1]:
            A = np.column_stack((A, np.zeros((A.shape[0], C.shape[1] - A.shape[1]))))

        # Create g with the label
        g = np.zeros(C.shape[1], dtype=np.float64)
        g[int(self.label - 10)] = 1


        self.max_bounds_value = np.abs(l_bounds.min()) if np.abs(l_bounds.min()) > np.abs(u_bounds.max()) else np.abs(u_bounds.max())

        return A, B, C, b, d, g

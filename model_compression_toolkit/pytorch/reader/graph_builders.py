# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import inspect
from typing import Dict, List, Tuple, Callable
import torch
from torch.fx import GraphModule

from model_compression_toolkit.common import BaseNode
from model_compression_toolkit.common.graph.base_graph import OutTensor
from model_compression_toolkit.common.graph.edge import Edge
from model_compression_toolkit.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.pytorch.constants import OUTPUT, PLACEHOLDER, TENSOR_META, CALL_FUNCTION, TYPE, \
    CALL_METHOD


class DummyPlaceHolder(object):
    """
    Class for PlaceHolder operator since a Pytorch model doesn't have one but FX does.
    """
    def __name__(self):
        return PLACEHOLDER


def nodes_builder(model: GraphModule,
                  module_dict: Dict,
                  to_numpy: Callable) -> Tuple[List, List, List, Dict]:
    """
    Build a node from a fx node. A node contains all information to reconstruct the model module or call function
    it's representing in the model: operation, module configuration, weights, input/output shape.
    Args:
        model: Pytorch FX model.
        module_dict: A dictionary of the Pyotrch model's named modules.

    Returns:
        A list of Graph nodes that were built from the fx GraphModule nodes.
    """
    # init function variables:
    inputs = []
    outputs = []
    nodes = []
    output_nodes = []
    fx_node_2_graph_node = {}

    for node in model.graph.nodes:
        # extract node type and framework attributes
        framework_attr = dict(node.kwargs)
        if node.target in module_dict.keys():
            node_type = type(module_dict[node.target])
            framework_attr = module_dict[node.target].__dict__
            fullargspec = inspect.getfullargspec(node_type.__init__).args
            framework_attr = {k: v for k, v in framework_attr.items() if k in fullargspec}
        elif node.op == CALL_FUNCTION:
            node_type = node.target
        elif node.op == PLACEHOLDER:
            node_type = DummyPlaceHolder
        elif node.op == OUTPUT:
            output_nodes += node.all_input_nodes
            continue
        elif node.op == CALL_METHOD:
            node_type = getattr(torch, node.target)
        else:
            raise Exception(f'Unknown node type: {node.name}')

        # extract layer weights and named buffers
        weights = {}
        if node.target in module_dict.keys():
            named_parameters_weights = {parameter[0]: to_numpy(parameter[1]) for parameter in
                                        module_dict[node.target].named_parameters()}
            named_buffer_weights = {parameter[0]: to_numpy(parameter[1]) for parameter in
                                    module_dict[node.target].named_buffers() if len(parameter[1].shape) > 0}
            weights.update(named_parameters_weights)
            weights.update(named_buffer_weights)

        # extract input shapes
        input_shape = []
        if node.op != PLACEHOLDER:
            for input_node in node.all_input_nodes:
                tensor_meta = input_node.meta
                if input_node.meta[TYPE] == torch.Tensor:
                    input_shape += [list(input_node.meta[TENSOR_META].shape)]
                elif input_node.meta[TYPE] == tuple:
                    input_shape += [list(n.shape) for n in input_node.meta[TENSOR_META]]

        # extract output shapes
        if node.meta[TYPE] == torch.Tensor:
            output_shape = [list(node.meta[TENSOR_META].shape)]
        elif node.meta[TYPE] == tuple:
            output_shape = [list(m.shape) for m in node.meta[TENSOR_META]]
        else:
            output_shape = []

        # if isinstance(node.meta[TENSOR_META], torch.fx.passes.shape_prop.TensorMetadata):
        #     output_shape = [list(node.meta[TENSOR_META].shape)]
        # else:
        #     output_shape = [list(m.shape) for m in node.meta[TENSOR_META]]

        # initiate graph nodes
        if node.op in [CALL_METHOD, CALL_FUNCTION]:
            graph_node_type = FunctionalNode
            inputs_as_list = len(node.args) > 0 and isinstance(node.args[0], (list, tuple)) and all(
                [isinstance(n, torch.fx.node.Node) for n in node.args[0]])
            num_inputs = 1 if inputs_as_list else len(node.all_input_nodes)
            op_call_args = list(node.args[num_inputs:])
            kwargs = {'functional_op': node_type,
                      'op_call_args': op_call_args,
                      'op_call_kwargs': node.kwargs,
                      'inputs_as_list': inputs_as_list}
        else:
            graph_node_type = BaseNode
            kwargs = {}
        graph_node = graph_node_type(name=node.name,
                                     framework_attr=framework_attr,
                                     input_shape=input_shape,
                                     output_shape=output_shape,
                                     weights=weights,
                                     layer_class=node_type,
                                     **kwargs)

        # generate graph inputs list
        if node.op == PLACEHOLDER:
            inputs.append(graph_node)

        fx_node_2_graph_node[node] = graph_node
        nodes.append(graph_node)

    # generate graph outputs list
    for node in output_nodes:
        outputs.append(OutTensor(fx_node_2_graph_node[node], output_nodes.index(node)))

    return nodes, inputs, outputs, fx_node_2_graph_node


def edges_builder(model: GraphModule,
                   fx_node_2_graph_node: Dict) -> List:
    """

    Args:
        model: Pytorch FX model.
        fx_node_2_graph_node: dictionary from fx node to graph node.

    Returns:
        List of graph edges
    """
    src_index = 0 # in fx src_index is always zero because fx uses the getitem operator to fetch node outputs
    edges = []
    connectivity_dict = {}
    for node in model.graph.nodes:
        if node.op != OUTPUT:
            for input_node in node.all_input_nodes:
                if connectivity_dict.get(input_node):
                    connectivity_dict[input_node].append((node, node.all_input_nodes.index(input_node)))
                else:
                    connectivity_dict[input_node] = [(node, node.all_input_nodes.index(input_node))]
    for node in model.graph.nodes:
        out_nodes = connectivity_dict.get(node)
        if out_nodes:
            for (out_node, dst_index) in out_nodes:
                edges.append(
                    Edge(fx_node_2_graph_node[node], fx_node_2_graph_node[out_node], src_index, dst_index))

    return edges
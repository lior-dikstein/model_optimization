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
import logging
from typing import Callable

import torch
from torch.nn import Module, BatchNorm2d
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from model_compression_toolkit.common import Graph, Node
from model_compression_toolkit.common.graph.base_graph import OutTensor
from model_compression_toolkit.common.graph.edge import Edge


def generate_module_dict(model):
    module_dict = dict()
    for m in model.named_modules():
        module_dict[m[0].replace('.', '_')] = m[1]
    return module_dict


def build_graph(model) -> Graph:
    """
    Given a Keras model, build and return an networkx MultiDiGraph containing all data (nodes, edges,
    inputs and outputs) representing that model.

    Args:
        model: Keras model to build its graph.

    Returns:
        Networkx MultiDiGraph representing the Keras model.

    """
    inputs = []
    outputs = []
    edges = []
    nodes = []
    fx_node_2_graph_node = {}
    output_index = 0
    module_dict = generate_module_dict(model)
    for node in model.graph.nodes:
        # if hasattr(model, node.name):
        #     node_type = type(getattr(model, node.name))
        #     framework_attr = getattr(model, node.name).__dict__
        #     fullargspec = inspect.getfullargspec(node_type.__init__).args
        #     framework_attr = {k: v for k, v in framework_attr.items() if k in fullargspec}
        if node.name in module_dict.keys():
            node_type = type(module_dict[node.name])
            framework_attr = module_dict[node.name].__dict__
            fullargspec = inspect.getfullargspec(node_type.__init__).args
            framework_attr = {k: v for k, v in framework_attr.items() if k in fullargspec}
        else:
            # print(node.name)
            node_type = node.target if node.op == 'call_function' else node.op
            framework_attr = node.kwargs


        if node.name in module_dict.keys():
            weights = {parameter[0]: parameter[1].detach().numpy() for parameter in module_dict[node.name].named_parameters()}
        else:
            weights = {}

        if node_type == BatchNorm2d:
            weights['running_mean'] = module_dict[node.name].running_mean.cpu().detach().numpy()
            weights['running_var'] = module_dict[node.name].running_var.cpu().detach().numpy()

        input_shape = []
        if node.op != 'placeholder':
            for input_node in node.all_input_nodes:
                tensor_meta = input_node.meta['tensor_meta']
                if isinstance(tensor_meta, torch.fx.passes.shape_prop.TensorMetadata):
                    input_shape += [list(tensor_meta.shape)]
                else:
                    input_shape += [list(n.shape) for n in tensor_meta]
            # input_shape = [list(n.meta['tensor_meta'].shape) for n in node.all_input_nodes]
        if isinstance(node.meta['tensor_meta'], torch.fx.passes.shape_prop.TensorMetadata):
            output_shape = [list(node.meta['tensor_meta'].shape)]
        else:
            output_shape = [list(m.shape) for m in node.meta['tensor_meta']]
        graph_node = Node(name=node.name.replace('_','.'),
                          framework_attr=framework_attr,
                          input_shape=input_shape,
                          output_shape=output_shape,
                          weights=weights,
                          layer_class=node_type,
                          op_call_args={'op_call_input_args': list(node.args[len(node.all_input_nodes):])}
                          )
        fx_node_2_graph_node[node] = graph_node
        if node.op == 'placeholder':
            inputs.append(graph_node)
        if node.op == 'output':
            outputs.append(OutTensor(graph_node, output_index))
            output_index += 1
        nodes.append(graph_node)
    for node in model.graph.nodes:
        dst_index = 0
        for out_node in model.graph.nodes:
            if len(out_node.all_input_nodes) > 0:
                if node in out_node.all_input_nodes:
                    src_index = 0#out_node.args.index(node)
                    edges.append(Edge(fx_node_2_graph_node[node], fx_node_2_graph_node[out_node], src_index, dst_index))
                    dst_index += 1
                elif isinstance(out_node.args[0], tuple) and node in out_node.args[0]:
                    src_index = out_node.args[0].index(node)
                    edges.append(Edge(fx_node_2_graph_node[node], fx_node_2_graph_node[out_node], src_index, dst_index))
                    dst_index += 1
    return Graph(nodes, inputs, outputs, edges)


def parse_model(model) -> Graph:
    """
    Parse a Keras model into a Graph.
    In case of a nested model, it recursively unrolls inner models.

    Args:
        model: Keras model to build its graph.

    Returns:
        Networkx MultiDiGraph representing the Keras model including: nodes, edges, inputs, and outputs.
    """
    model_graph = build_graph(model)

    return model_graph


def fx_graph_module_generation(pytorch_model, representative_data_gen):
    pytorch_model.eval()
    symbolic_traced = symbolic_trace(pytorch_model)
    ShapeProp(symbolic_traced).propagate(*representative_data_gen())
    return symbolic_traced


def model_reader(model: Module, representative_data_gen: Callable) -> Graph:
    """
    Reads a Pytorch model, converts it to an FX Graph using the fx toolkit, then builds a base graph representing the fx graph.
    Args:
        model: Pytorch model to build its graph representation.
        representative_data_gen (Callable): Dataset used for calibration.

    Returns:
        Base graph of the Pytorch model.
    """
    logging.info("Start Model Reading...")
    fx_model = fx_graph_module_generation(model, representative_data_gen)
    graph = parse_model(fx_model)
    return graph
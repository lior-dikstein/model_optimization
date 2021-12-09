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
from typing import Tuple, Any, Dict, List

import torch
from networkx import topological_sort

from model_compression_toolkit import common, FrameworkInfo
from model_compression_toolkit.common import Node, Graph
from model_compression_toolkit.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.pytorch.back2framework.instance_builder import node_builder
from model_compression_toolkit.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO


def build_input_tensors_list(node: Node,
                             graph: Graph,
                             node_to_output_tensors_dict: Dict[Node, List]) -> List[List]:
    """
    Given a node, build a list of input tensors the node gets. The list is built
    based on the node's incoming edges and previous nodes' output tensors.

    Args:
        node: Node to build its input tensors list.
        graph: Graph the node is in.
        node_to_output_tensors_dict: A dictionary from a node to its output tensors.

    Returns:
        A list of the node's input tensors.
    """

    input_tensors = []
    # Go over a sorted list of the node's incoming edges, and for each source node get its output tensors.
    # Append them in a result list.
    for ie in graph.incoming_edges(node, sort_by_attr=EDGE_SINK_INDEX):
        _input_tensors = [node_to_output_tensors_dict[ie.source_node][ie.source_index]]
        input_tensors.append(_input_tensors)
    return input_tensors


def run_operation(n: Node,
                  input_tensors,
                  op_func,
                  placeholder_output: torch.Tensor,
                  quantized: bool = True):
    """
    Applying the layer (op_func) to the input tensors (input_tensors).
    If quantized is set to True, and the layer's corresponding node (n) has quantization
    attributes, an additional fake-quantization node is built and appended to the layer.

    Args:
        n: The corresponding node of the layer it runs.
        input_tensors: List of references to Keras tensors that are the layer's inputs.
        op_func: Layer to apply to the input tensors.
        placeholder_output:

    Returns:
        A list of references to Keras tensors. The layer's output tensors after applying the
        layer to the input tensors.
    """

    if len(input_tensors) == 0:  # Placeholder handling
        out_tensors_of_n = placeholder_output
        if quantized:  # Add a fake quant node
            fake_quant = n.activation_quantization_cfg.activation_quantization_fn(
                n.activation_quantization_cfg.activation_n_bits,
                n.activation_quantization_cfg.activation_is_signed,
                n.activation_quantization_cfg.activation_quantization_params)
            if fake_quant is not None:
                out_tensors_of_n = fake_quant(out_tensors_of_n)

    else:
        input_tensors = [tensor for tensor_list in input_tensors for tensor in tensor_list]  # flat list of lists

        # If operator expects a single input tensor, it cannot be a list as it should
        # have a dtype field.
        # out_tensors_of_n = op_func(*input_tensors + n.op_call_input_args, **n.op_call_args)
        out_tensors_of_n = op_func(*input_tensors + n.op_call_args['op_call_input_args'])

        # Add a fake quant node if the node has an activation threshold.
        if quantized and n.activation_quantization_cfg is not None and n.activation_quantization_cfg.enable_activation_quantization:
            fake_quant = n.activation_quantization_cfg.activation_quantization_fn(
                n.activation_quantization_cfg.activation_n_bits,
                n.activation_quantization_cfg.activation_is_signed,
                n.activation_quantization_cfg.activation_quantization_params)
            if fake_quant is not None:
                out_tensors_of_n = fake_quant(out_tensors_of_n)

    return out_tensors_of_n


class BackToPytorch(torch.nn.Module):
    def __init__(self, graph: Graph,
                 mode: ModelBuilderMode = ModelBuilderMode.QUANTIZED,
                 append2output: List[Any] = None):
        super(BackToPytorch, self).__init__()
        self.graph = graph
        self.mode = mode
        self.node_sort = list(topological_sort(graph))
        self.nodes_dict = {}
        self.append2output = append2output
        for n in self.node_sort:
            if inspect.isclass(n.layer_class) and issubclass(n.layer_class, torch.nn.Module):
                setattr(self, n.name, node_builder(n))


    def forward(self, inputs):
        node_to_output_tensors_dict = dict()
        for n in self.node_sort:
            if n.layer_class != 'output':
                placeholder_output = inputs[self.graph.get_inputs().index(n)] if n.layer_class == 'placeholder' else []
                input_tensors = build_input_tensors_list(n,
                                                         self.graph,
                                                         node_to_output_tensors_dict)
                out_tensors_of_n = run_operation(n,  # Run node operation and fetch outputs
                                                 input_tensors,
                                                 getattr(self, n.name) if hasattr(self, n.name) else n.layer_class,
                                                 placeholder_output,
                                                 quantized=self.mode == ModelBuilderMode.QUANTIZED)
                # quantized=self.mode)
                if isinstance(out_tensors_of_n, list):
                    node_to_output_tensors_dict.update({n: out_tensors_of_n})
                else:
                    node_to_output_tensors_dict.update({n: [out_tensors_of_n]})

        if self.append2output:
            output = []
            for n in self.append2output:
                output.append(node_to_output_tensors_dict.get(n)[0])
            return output
        else:
            return [v[0] for v in node_to_output_tensors_dict.values()]


def model_builder(graph: common.Graph,
                  mode: ModelBuilderMode = ModelBuilderMode.QUANTIZED,
                  append2output: List[Any] = None,
                  fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO) -> Tuple[torch.nn.Module, Any]:
    """
    Build a Pytorch model from a graph representing the model.
    The model is built by converting the graph nodes to torch modules and applying them sequentially to get the model
    output tensors. The output tensors list and an input tensors list, are then used to build the model.
    When the model is not built in float mode, the graph is transformed by additional substitutions.

    Args:
        graph: Graph to build its corresponding Pytorch model.
        mode: Building mode. Read ModelBuilderMode description for more info.
        append2output: List of nodes or OutTensor objects. In float building mode,
        when the list contains nodes, all output tensors of all nodes are set as the model outputs.
        fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).
        This is needed when using MIXEDPRECISION or GPTQ mode for passing the kernel attributes to
        the QuanteWrapper we use in both of these cases.

    Returns:
        A tuple of the model, and an UserInformation object.
    """

    model = BackToPytorch(graph, mode, append2output)

    return model, graph.user_info
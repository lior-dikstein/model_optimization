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

import torch

from model_compression_toolkit.common import Node


def build_node(model,
               node,
               node_name_to_node: dict) -> Node:
    """
    Build a node from a Keras node. A node contains all information to reconstruct the layer it's representing
    in a model:
    operation, layer configuration, path for instantiating the Keras layer the node has, weights, group of other
    nodes if it's a reused layer,
    input/output shape.
    Args:
        node: Node in the graph of a Keras model.
        node_name_to_node: Dictionary of already created nodes aims to identify reused layers.

    Returns:
        Graph node that was built from the Keras node.
    """
    if hasattr(model, node.name):
        node_type = type(getattr(model, node.name))
        framework_attr = getattr(model, node.name).__dict__
        fullargspec = inspect.getfullargspec(node_type.__init__).args
        framework_attr = {k: v for k, v in framework_attr.items() if k in fullargspec}
    else:
        node_type = node.op
        framework_attr = node.__dict__

    if hasattr(model, node.name):
        weights = {parameter[0]: parameter[1].detach().numpy() for parameter in getattr(model, node.name).named_parameters()}
    else:
        weights = {}
    if node.op == 'placeholder':
        input_shape = []
    else:
        input_shape = []
        for n in node.all_input_nodes:
            if isinstance(n.meta['tensor_meta'], torch.fx.passes.shape_prop.TensorMetadata):
                input_shape = [list(n.meta['tensor_meta'].shape)]
            else:
                input_shape = input_shape + [list(m.shape) for m in n.meta['tensor_meta']]
    if isinstance(node.meta['tensor_meta'], torch.fx.passes.shape_prop.TensorMetadata):
        output_shape = [list(node.meta['tensor_meta'].shape)]
    else:
        output_shape = [list(m.shape) for m in node.meta['tensor_meta']]
    graph_node = Node(name=node.name,
                      framework_attr=framework_attr,
                      input_shape=input_shape,
                      output_shape=output_shape,
                      weights=weights,
                      layer_class=node_type)

    node_name_to_node[node.name] = graph_node

    return graph_node
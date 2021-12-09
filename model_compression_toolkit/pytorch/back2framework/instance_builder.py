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
import copy
from typing import List

import torch

from model_compression_toolkit import common
from model_compression_toolkit.common import Node


def node_builder(n: common.Node):
    """
    Build a Pytorch module from a node.

    Args:
        n: Node to build its Keras layer

    Returns:
        Pytorch module that was built from the node.
    """

    framework_attr = copy.copy(n.framework_attr)
    node_instance = n.layer_class(**framework_attr)
    for k, v in n.weights.items():
        setattr(node_instance, k, torch.nn.Parameter(torch.Tensor(v), requires_grad=False))
    node_instance.trainable = False  # Set all node as not trainable
    return node_instance


def instance_builder(toposort: List[Node]):
    """
    Build a dictionary of nodes to their corresponding Keras
    layers, given a list of nodes.

    Args:
        toposort: List of nodes sorted topological to build their layers.

    Returns:
        A dictionary of nodes to their corresponding Keras layers.
    """

    nodes_dict = dict()
    for n in toposort:
        if not n.reuse:
            if not type(n.layer_class) == str:# Hold a single node in dictionary for all reused nodes from the same layer.
                nodes_dict.update({n: node_builder(n)})
        print()
    return nodes_dict
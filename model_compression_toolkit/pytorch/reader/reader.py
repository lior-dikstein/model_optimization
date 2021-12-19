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


import logging
from typing import Callable, Dict
import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from model_compression_toolkit.common import Graph
from model_compression_toolkit.pytorch.reader.graph_builders import edges_builder, nodes_builder


def generate_module_dict(model: torch.nn.Module) -> Dict:
    """
    Creates a dictionary from the Pytorch model's named modules by module name.

    Args:
        model: A Pytorch model.

    Returns:
        A dictionary of the Pyotrch model's named modules.
    """
    module_dict = dict()
    for m in model.named_modules():
        module_dict[m[0]] = m[1]
    return module_dict


def build_graph(model: torch.fx.GraphModule) -> Graph:
    """
    Given a Pytorch FX model, build and return an networkx MultiDiGraph containing all data (nodes, edges,
    inputs and outputs) representing that model.

    Args:
        model: Pytorch FX model to build its graph.

    Returns:
        Networkx MultiDiGraph representing the Keras model.

    """
    # generate a dictionary with all model modules
    module_dict = generate_module_dict(model)

    # convert fx nodes to generic graph nodes
    nodes, inputs, outputs, fx_node_2_graph_node = nodes_builder(model, module_dict)

    # build graph edges
    edges = edges_builder(model, fx_node_2_graph_node)

    return Graph(nodes, inputs, outputs, edges)


def fx_graph_module_generation(pytorch_model: torch.nn.Module,
                               representative_data_gen: Callable) -> torch.fx.GraphModule:
    """
    Generates a fx.GraphModule from a torch.nn.Module.

    Args:
        pytorch_model: A dynamic Pytorch model.
        representative_data_gen (Callable): Representative dataset used for shape inference.

    Returns:
        A fx.GraphModule (static model) representing the Pytorch model.
    """
    pytorch_model.eval()
    symbolic_traced = symbolic_trace(pytorch_model)
    ShapeProp(symbolic_traced).propagate(*representative_data_gen())
    return symbolic_traced


def model_reader(model: torch.nn.Module,
                 representative_data_gen: Callable) -> Graph:
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
    graph = build_graph(fx_model)
    return graph
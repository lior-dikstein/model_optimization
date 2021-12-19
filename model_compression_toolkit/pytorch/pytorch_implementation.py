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


from typing import List, Any, Tuple, Callable
import numpy as np
import torch
from torch.nn import Module

from model_compression_toolkit import QuantizationConfig, FrameworkInfo, common, GradientPTQConfig, \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.common import Graph, BaseNode
from model_compression_toolkit.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.common.user_info import UserInformation
from model_compression_toolkit.pytorch.back2framework.model_builder import model_builder
from model_compression_toolkit.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.pytorch.graph_substitutions.substitutions.batchnorm_folding import \
    BatchNormalizationFolding
from model_compression_toolkit.pytorch.reader.reader import model_reader
from model_compression_toolkit.pytorch.tensor_marking import get_node_stats_collector
import model_compression_toolkit.pytorch.constants as pytorch_constants


class PytorchImplementation(FrameworkImplementation):
    """
    An class with implemented methods to support optimizing Pytorch models.
    """

    def __init__(self):
        super().__init__()

    @property
    def constants(self):
        """

        Returns: Module of Pytorch constants.

        """
        return pytorch_constants

    def model_reader(self,
                     module: Module,
                     representative_data_gen: Callable) -> Graph:
        """
        Convert a framework's module into a graph.
        Args:
            module: Framework's module.
            representative_data_gen (Callable): Dataset used for calibration.

        Returns:
            Graph representing the input module.
        """
        return model_reader(module, representative_data_gen)

    def to_numpy(self,
                 tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a Pytorch tensor to a Numpy array.
        Args:
            tensor: Pytorch tensor.

        Returns:
            Numpy array converted from the input tensor.
        """
        return tensor.detach().cpu().numpy()

    def model_builder(self,
                      graph: Graph,
                      mode: ModelBuilderMode,
                      append2output: List[Any] = None,
                      fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO) -> Tuple[Module, UserInformation]:
        """
        Build a Pytorch module from a graph.
        The mode determines how the module should be build. append2output is a list of Nodes
        to set as the module outputs.

        Args:
            graph: Graph to build the module from it.
            mode: Mode for how to build the module.
            append2output: List of Nodes to set as the module's outputs.
            fw_info: FrameworkInfo object with information about the specific framework's module

        Returns:
            A tuple of the Pytorch module that was built and an UserInformation object.
        """
        return model_builder(graph,
                             mode,
                             append2output,
                             fw_info)

    def shift_negative_correction(self,
                                  graph: Graph,
                                  qc: QuantizationConfig,
                                  fw_info: FrameworkInfo) -> Graph:
        """
        Apply shift negative correction (SNC) on a graph.

        Args:
            graph: Graph to apply SNC on.
            qc: Quantization configuration.
            fw_info: FrameworkInfo object with information about the specific framework's module.

        Returns:
            Graph after SNC.
        """
        raise Exception('This feature is currently not yet available for Pytorch models. Work in progress.')

    def attach_sc_to_node(self,
                          node: BaseNode,
                          fw_info: FrameworkInfo) -> common.statistics_collector.BaseStatsContainer:
        """
        Return a statistics collector that should be attached to a node's output
        during statistics collection.

        Args:
            node: Node to return its collector.
            fw_info: FrameworkInfo object with information about the specific framework's module

        Returns:
            Statistics collector for the node.
        """
        return get_node_stats_collector(node,
                                        fw_info)

    def get_substitutions_marking(self) -> List[common.BaseSubstitution]:
        """

        Returns: A list of the framework substitutions used for marking
        points we fuse.

        """
        return []

    def get_substitutions_pre_statistics_collection(self) -> List[common.BaseSubstitution]:
        """

        Returns: A list of the framework substitutions used before we build a quantized module.

        """
        return [BatchNormalizationFolding()]

    def get_substitutions_post_statistics_collection(self,
                                                     quant_config: QuantizationConfig) -> List[common.BaseSubstitution]:
        """
        Return a list of the framework substitutions used after we collect statistics.

        Args:
            quant_config: QuantizationConfig to determine which substitutions to return.

        Returns:
            A list of the framework substitutions used after we collect statistics.
        """
        substitutions_list = []
        return substitutions_list

    def get_substitutions_channel_equalization(self,
                                               quant_config: QuantizationConfig,
                                               fw_info: FrameworkInfo) -> List[common.BaseSubstitution]:
        """
        Return a list of the framework substitutions used for channel equalization.

        Args:
            quant_config: QuantizationConfig to determine which substitutions to return.
            fw_info: FrameworkInfo object with information about the specific framework's module.

        Returns:
            A list of the framework substitutions used after we collect statistics.
        """
        substitutions_list = []
        return substitutions_list

    def get_substitutions_pre_build(self) -> List[common.BaseSubstitution]:
        """

        Returns: A list of the framework substitutions used before we build a quantized module.

        """
        return []

    def gptq_training(self,
                      graph: Graph,
                      representative_data_gen: Callable,
                      gptq_config: GradientPTQConfig,
                      fw_info: FrameworkInfo) -> Graph:
        """
        Update a graph using GPTQ after minimizing the loss between the float module's output
        and the quantized module's outputs.

        Args:
            graph: Graph to fine-tune.
            representative_data_gen: Dataset to use for inputs of the models.
            gptq_config: GradientPTQConfig with configuration for the fine-tuning process.
            fw_info: FrameworkInfo object with information about the specific framework's module.

        Returns:
            Updated graph after GPTQ.
        """
        raise Exception('This feature is currently not yet available for Pytorch models. Work in progress.')

    def get_sensitivity_evaluation_fn(self,
                                      graph: Graph,
                                      quant_config: MixedPrecisionQuantizationConfig,
                                      metrics_weights: np.ndarray,
                                      representative_data_gen: Callable,
                                      fw_info: FrameworkInfo) -> Callable:
        """
        Create and return a function to compute a sensitivity metric for a mixed-precision
        configuration (comparing to the float Pytorch module).

        Args:
            graph: Graph to build it's float and mixed-precision Pytorch models.
            quant_config: QuantizationConfig of how the module should be quantized.
            metrics_weights: Array of weights to weight the sensitivity among different layers.
            representative_data_gen: Dataset to use for retrieving images for the models inputs.
            fw_info: FrameworkInfo object with information about the specific framework's module.

        Returns:
            A function that computes the metric.
        """
        raise Exception('This feature is currently not yet available for Pytorch models. Work in progress.')
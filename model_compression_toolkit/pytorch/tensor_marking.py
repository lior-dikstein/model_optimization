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
from model_compression_toolkit import common, FrameworkInfo
from model_compression_toolkit.common.statistics_collector import BaseStatsContainer


def get_stats_collector(n: common.Node,
                        fw_info: FrameworkInfo) -> BaseStatsContainer:
    """
    Create and initial a statistics collector for a linear operator. If the layer has an activation function and
    its min/max output values are known, the statistics collector is initialized with these values.
    If the layer's output should not be quantized, NoStatsContainer is created.

    Args:
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)
        n: Node to create a statistics collector for it.

    Returns:
        BaseStatsContainer according to statistics that are collected.
    """
    if n.layer_class in fw_info.layer_min_max_mapping.keys():
        min_value, max_value = fw_info.layer_min_max_mapping[n.layer_class]
        return common.StatsContainer(init_min_value=min_value,
                                     init_max_value=max_value)
    if n.output_quantization:
        return common.StatsContainer()
    else:
        return common.NoStatsContainer()


def get_node_stats_collector(node: common.Node,
                             fw_info: common.FrameworkInfo) -> common.statistics_collector.BaseStatsContainer:
    """
    Gets a node and a groups list and create and return a statistics collector for the node
    according to the group the node is in.

    Args:
        node: Node to create its statistics collector.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)

    Returns:
        Statistics collector for statistics collection for the node.
    """

    stats_collector = get_stats_collector(node, fw_info)
    if fw_info.in_no_quantization_ops(node):  # node should not be quantized
        stats_collector = common.NoStatsContainer()

    return stats_collector
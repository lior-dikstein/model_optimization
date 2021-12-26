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


import unittest
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet

from model_compression_toolkit import DEFAULTCONFIG
from model_compression_toolkit.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.keras.reader.reader import model_reader
from model_compression_toolkit.common.model_collector import ModelCollector
from model_compression_toolkit.common.quantization.quantization_analyzer import analyzer_graph
from model_compression_toolkit.common.collectors.statistics_collector_generator import create_stats_collector_for_node
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO

class TestGraphStatisticCollection(unittest.TestCase):

    def test_bn_folding_mbv1(self):
        model = MobileNet()
        graph = model_reader(model)  # model pharsing
        fw_impl = KerasImplementation()
        tg = substitute(graph, fw_impl.get_substitutions_pre_statistics_collection())  # substition
        for node in tg.nodes:
            node.prior_info = fw_impl.get_node_prior_info(node=node,
                                                          fw_info=DEFAULT_KERAS_INFO)

        tg = set_quantization_configuration_to_graph(graph=tg,
                                                     quant_config=DEFAULTCONFIG,
                                                     fw_info=DEFAULT_KERAS_INFO)

        analyzer_graph(create_stats_collector_for_node,
                       tg,
                       DEFAULT_KERAS_INFO)  # Mark point for collection of statistics

        mi = ModelCollector(tg, fw_impl, DEFAULT_KERAS_INFO)
        for i in range(10):
            mi.infer([np.random.randn(1, 224, 224, 3)])


if __name__ == '__main__':
    unittest.main()

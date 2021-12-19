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
import random
import unittest

from model_compression_toolkit.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from tests.keras_tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
import model_compression_toolkit as mct
import torch
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity
import torchvision.models as models

class PytorchTest(BaseFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_configs(self):
        return {'all_32bit': mct.QuantizationConfig(mct.ThresholdSelectionMethod.NOCLIPPING,
                                                    mct.ThresholdSelectionMethod.NOCLIPPING,
                                                    mct.QuantizationMethod.POWER_OF_TWO,
                                                    mct.QuantizationMethod.POWER_OF_TWO,
                                                    32, 32, False, False, True,
                                                    enable_weights_quantization=True,
                                                    enable_activation_quantization=True),
                'all_4bit': mct.QuantizationConfig(mct.ThresholdSelectionMethod.NOCLIPPING,
                                                   mct.ThresholdSelectionMethod.NOCLIPPING,
                                                   mct.QuantizationMethod.POWER_OF_TWO,
                                                   mct.QuantizationMethod.POWER_OF_TWO,
                                                   4, 4, False, False, True,
                                                   enable_weights_quantization=True,
                                                   enable_activation_quantization=True),
                'weights_4bit': mct.QuantizationConfig(mct.ThresholdSelectionMethod.NOCLIPPING,
                                                       mct.ThresholdSelectionMethod.NOCLIPPING,
                                                       mct.QuantizationMethod.POWER_OF_TWO,
                                                       mct.QuantizationMethod.POWER_OF_TWO,
                                                       4, 4, False, False, True,
                                                       enable_weights_quantization=True,
                                                       enable_activation_quantization=False),
                'activations_4bit': mct.QuantizationConfig(mct.ThresholdSelectionMethod.NOCLIPPING,
                                                           mct.ThresholdSelectionMethod.NOCLIPPING,
                                                           mct.QuantizationMethod.POWER_OF_TWO,
                                                           mct.QuantizationMethod.POWER_OF_TWO,
                                                           4, 4, False, False, True,
                                                           enable_weights_quantization=False,
                                                           enable_activation_quantization=True)
                }

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 224, 224]]

    @staticmethod
    def generate_inputs(input_shapes):
        return [torch.randn(*in_shape) for in_shape in input_shapes]

    def create_feature_network(self, input_shape):
        pass

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        float_model.eval()
        float_result = float_model(*input_x)
        for model_name, quantized_model in quantized_models.items():
            quantized_model.eval()
            quant_result = quantized_model(input_x)
            for i, (f, q) in enumerate(zip(float_result, quant_result)):
                # print(f'Float values: max - {f.abs().max().item()}, {q.abs().max().item()}')
                print(f'{model_name} output {i} error: max - {np.max(np.abs(f.cpu().detach().numpy() - q.cpu().detach().numpy()))}, sum - {np.sum(np.abs(f.cpu().detach().numpy() - q.cpu().detach().numpy()))}')
        self.unit_test.assertTrue(True)

    def run_test(self, seed=0):
        np.random.seed(seed)
        random.seed(a=seed)
        torch.random.manual_seed(seed)
        input_shapes = self.create_inputs_shape()
        x = self.generate_inputs(input_shapes)

        def representative_data_gen():
            return x

        ptq_models = {}
        model_float = self.create_feature_network(input_shapes)
        for model_name, quant_config in self.get_quantization_configs().items():
            ptq_model, quantization_info = mct.pytorch_post_training_quantization(model_float,
                                                                                  representative_data_gen,
                                                                                  n_iter=1,
                                                                                  quant_config=quant_config,
                                                                                  fw_info=DEFAULT_PYTORCH_INFO,
                                                                                  network_editor=self.get_network_editor())
            ptq_models.update({model_name: ptq_model})
        self.compare(ptq_models, model_float, input_x=x, quantization_info=quantization_info)

class RunPytorchTest(unittest.TestCase):
    def test_pytorch_quantizer(self):
        PytorchTest(self).run_test()


if __name__ == '__main__':
    unittest.main()
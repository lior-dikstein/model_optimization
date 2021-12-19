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
import model_compression_toolkit as mct
import torch
import numpy as np
import torchvision.models as models

from tests.pytorch_tests.feature_networks_tests.feature_networks.base_module_test import PytorchTest
from tests.pytorch_tests.utils.utils import IMG_NET_VAL_PREPROCESSED_IMAGE_00000003


class MobileNetV2Test(PytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)


    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 224, 224]]


    @staticmethod
    def generate_inputs(input_shapes):
        import pickle
        with open('../../' + IMG_NET_VAL_PREPROCESSED_IMAGE_00000003, 'rb') as p:
            input_batch = pickle.load(p)
        return [input_batch]

    def create_feature_network(self, input_shape):
        return models.mobilenet_v2(pretrained=True)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        float_model.eval()
        float_result = float_model(*input_x)
        float_probabilities = torch.nn.functional.softmax(float_result[0], dim=0)
        for model_name, quantized_model in quantized_models.items():
            quantized_model.eval()
            quant_result = quantized_model(input_x)
            quant_probabilities = torch.nn.functional.softmax(quant_result[0][0], dim=0)
            for i, (f, q) in enumerate(zip(float_result, quant_result)):
                print(f'{model_name} output {i} error: max - {np.max(np.abs(f.cpu().detach().numpy() - q.cpu().detach().numpy()))}, sum - {np.sum(np.abs(f.cpu().detach().numpy() - q.cpu().detach().numpy()))}')
                print(f'{model_name} output {i} probabilities: float - {float_probabilities.max().item()}, quant - {quant_probabilities.max().item()}')
        self.unit_test.assertTrue(True)


class RunMobileNetV2Test(unittest.TestCase):
    def test_pytorch_quantizer(self):
        MobileNetV2Test(self).run_test()


if __name__ == '__main__':
    unittest.main()
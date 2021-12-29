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

import torch
from tests.pytorch_tests.feature_networks_tests.feature_networks.base_module_test import PytorchTest


class AddNet(torch.nn.Module):
    def __init__(self):
        super(AddNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=1, stride=1)

    def forward(self, x, y):
        x = self.conv1(x)
        y = self.conv2(y)
        return x - y, y - x, x + y


class AddNetTest(PytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 224, 224], [self.val_batch_size, 3, 224, 224]]

    @staticmethod
    def generate_inputs(input_shapes):
        return [torch.randn(*in_shape) for in_shape in input_shapes]

    def create_feature_network(self, input_shape):
        return AddNet()


class RunPytorchTest(unittest.TestCase):
    def test_pytorch_quantizer(self):
        AddNetTest(self).run_test()


if __name__ == '__main__':
    unittest.main()
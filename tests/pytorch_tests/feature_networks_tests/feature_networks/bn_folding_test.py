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


class BNFoldingNet(torch.nn.Module):
    def __init__(self):
        super(BNFoldingNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        return x


class BNFoldingNetTest(PytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 224, 224]]

    def create_feature_network(self, input_shape):
        return BNFoldingNet()


class RunSplitNetTestTest(unittest.TestCase):
    def test_pytorch_quantizer(self):
        BNFoldingNetTest(self).run_test()


if __name__ == '__main__':
    unittest.main()
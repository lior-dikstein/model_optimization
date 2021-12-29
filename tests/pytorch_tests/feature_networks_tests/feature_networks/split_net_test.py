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


class SplitNet(torch.nn.Module):
    def __init__(self):
        super(SplitNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(1, 3, kernel_size=1, stride=1)
        self.conv3 = torch.nn.Conv2d(1, 3, kernel_size=1, stride=1)

    def forward(self, x, y):
        x = self.conv1(x)
        y = self.conv1(y)
        x1, x2, x3 = torch.split(x, split_size_or_sections=1, dim=1)
        y1, y2, y3 = torch.split(y, split_size_or_sections=1, dim=1)
        x4 = (x3 - x1) * x2
        xy1 = torch.concat([x2, y2], 1)
        xy2 = torch.concat([x1, y3], 1)
        return self.conv3(x3), self.conv2(x2), x4, xy1 - xy2


class SplitNetTest(PytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 224, 224], [self.val_batch_size, 3, 224, 224]]

    def create_feature_network(self, input_shape):
        return SplitNet()


class RunSplitNetTestTest(unittest.TestCase):
    def test_pytorch_quantizer(self):
        SplitNetTest(self).run_test()


if __name__ == '__main__':
    unittest.main()
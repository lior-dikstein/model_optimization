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



class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(4, 4, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(4, 4, kernel_size=3, stride=2)
        self.bn = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU()



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        # x = self.relu(x) + x
        # x1, x2, x3 = torch.split(x, split_size_or_sections=2)
        # x4 = (x1 - x2) * x3
        # return self.conv2(x), self.conv3(y), x4
        # return self.conv2(x)
        return x

    # def forward(self, x):
    #     x = self.conv1(x)
    #     # y = self.conv1(y)
    #     x = self.relu(x)
    #     # y = self.relu(y)
    #     return self.conv2(x), self.conv3(x) #+ 77*self.conv3(x).clamp(min=0., max=1.)


# class MyModule(torch.nn.Module):
#     def __init__(self):
#         super(MyModule, self).__init__()
#         self.conv = torch.nn.Conv2d(4, 4, kernel_size=1, stride=1)
#
#
#
#     def forward(self, x):
#         x = x + x
#         return self.conv(88 + x)

class PytorchTest(BaseFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.ThresholdSelectionMethod.NOCLIPPING,
                                      mct.ThresholdSelectionMethod.NOCLIPPING,
                                      mct.QuantizationMethod.POWER_OF_TWO,
                                      mct.QuantizationMethod.KMEANS,
                                      4, 4, False, False, True,
                                      enable_weights_quantization=True,
                                      enable_activation_quantization=True)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 224, 224]]
        # return [[self.val_batch_size, 3, 224, 224], [self.val_batch_size, 4, 224, 224]]

    @staticmethod
    def generate_inputs(input_shapes):
        filename = '/Vols/vol_design/tools/swat/datasets_src/ImageNet/ILSVRC2012_img_val/ILSVRC2012_val_00000003.JPEG'
        from PIL import Image
        from torchvision import transforms
        input_image = Image.open(filename)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        return [input_batch]
        # import pickle
        # with open('/Vols/vol_design/tools/swat/users/liord/datasets/image_244_pickle.pickle', 'rb') as p:
        #     data = pickle.load(p)
        # return [torch.from_numpy(data['data'][:input_shapes[0][0], :, :, :].transpose(0, 3, 1, 2).astype(np.float32))]

    def create_feature_network(self, input_shape):
        # return MyModule()
        return models.mobilenet_v2(pretrained=True)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        input_shapes = self.create_inputs_shape()
        x = self.generate_inputs(input_shapes)
        # y = quantized_model(x)
        # float_model_params_dict = {param[0]: param[1].detach().numpy() for param in float_model.named_parameters()}
        # quant_model_params_dict = {param[0]: param[1].detach().numpy() for param in quantized_model.named_parameters()}
        # for k, v in float_model_params_dict.items():
        #     print(k, np.sum(np.abs(v - quant_model_params_dict.get(k))))
        # import matplotlib.pyplot as plt
        # plt.imshow(x[0].numpy()[0, :, :, :].transpose(1, 2, 0))
        # plt.show()
        activation = {}

        def get_activation(name):
            def hook(float_model, input, output):
                activation[name] = output.detach()

            return hook

        for m in float_model.named_modules():
            if isinstance(m[1], torch.nn.Conv2d) or isinstance(m[1], torch.nn.BatchNorm2d) or isinstance(m[1], torch.nn.ReLU6):
                m[1].register_forward_hook(get_activation(m[0]))
        float_model.eval()
        quantized_model.eval()
        float_result = float_model(x[0])
        quant_result = quantized_model(x)
        print(f'Float values: max - {np.max(np.abs(float_result.cpu().detach().numpy()))}')
        print(f'Quant values: max - {np.max(np.abs(quant_result[-1].cpu().detach().numpy()))}')
        print(f'Resulting output error: sum - {np.sum(np.abs(float_result.cpu().detach().numpy() - quant_result[-1].cpu().detach().numpy()))}, mean - {np.mean(np.abs(float_result.cpu().detach().numpy() - quant_result[-1].cpu().detach().numpy()))}')
        print(f'Resulting output error: max - {np.max(np.abs(float_result.cpu().detach().numpy() - quant_result[-1].cpu().detach().numpy()))}, min - {np.min(np.abs(float_result.cpu().detach().numpy() - quant_result[-1].cpu().detach().numpy()))}')
        self.unit_test.assertTrue(True)

    def run_test(self, seed=0):
        np.random.seed(seed)
        random.seed(a=seed)
        torch.random.manual_seed(seed)
        input_shapes = self.create_inputs_shape()
        x = self.generate_inputs(input_shapes)

        def representative_data_gen():
            return x

        model_float = self.create_feature_network(input_shapes)
        ptq_model, quantization_info = mct.pytorch_post_training_quantization(model_float,
                                                                              representative_data_gen,
                                                                              n_iter=1,
                                                                              quant_config=self.get_quantization_config(),
                                                                              fw_info=DEFAULT_PYTORCH_INFO,
                                                                              network_editor=self.get_network_editor())
        self.compare(ptq_model, model_float, input_x=x, quantization_info=quantization_info)

class RunPytorchTest(unittest.TestCase):
    def test_pytorch_quantizer(self):
        PytorchTest(self).run_test()


if __name__ == '__main__':
    unittest.main()
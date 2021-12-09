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



from typing import Tuple, Callable

import numpy as np
import torch

from model_compression_toolkit.common.constants import THRESHOLD


def quantizer_min_max_calculator(threshold: np.ndarray,
                                 num_bits: int,
                                 signed: bool) -> Tuple[float, float]:
    """
    Compute quantization range's min/max values given a threshold, number of bits,
     and whether it's signed or not.

    Args:
        threshold: Threshold for quantization range values.
        num_bits: Number of bits to use for quantization.
        signed: Whether the quantization range should include negative values or not.

    Returns:
        Min and max values for quantization range.
    """

    if signed:
        delta = threshold / (2 ** (num_bits - 1))
        min_value = -threshold
    else:
        delta = threshold / (2 ** (num_bits))
        min_value = 0

    max_value = threshold - delta
    return min_value, max_value


def constraint_quantization(activation_n_bits: int,
                            activation_is_signed: bool,
                            quantization_params: dict) -> Callable:
    """
    Use a NodeQuantizationConfig to compute a quantizer min/max values, and use it to
    build and return a fake-quantization node.

    Args:
        activation_n_bits: Number of bits to use for quantization.
        activation_is_signed: Whether the quantization range should include negative values or not.
        quantization_params: Dictionary of specific parameters for this quantization function.

    Returns:
        A fake quantization node.
    """
    activation_threshold = quantization_params.get(THRESHOLD)
    if activation_threshold is None:
        return None

    min_value, max_value = quantizer_min_max_calculator(activation_threshold,
                                                        activation_n_bits,
                                                        activation_is_signed)

    def q(x: torch.Tensor) -> torch.Tensor:
        """
        Fake-quantize the input tensor x, using a pytorch fake-quantization node.

        Args:
            x: Input tensor to quantize.

        Returns:
            The fake-quantized input tensor.
        """
        scale = 1 / 2**(activation_n_bits - 1)
        return torch.fake_quantize_per_tensor_affine(x,
                                                     scale=scale,
                                                     zero_point=0,
                                                     quant_min=int(min_value / scale),
                                                     quant_max=int(max_value / scale))

    return q

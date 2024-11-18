import idx2numpy
import numpy as np
from scipy.signal import convolve2d as conv2d
import sys
import os
import subprocess

from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model
from onnx.numpy_helper import from_array

I = make_tensor_value_info('I', TensorProto.FLOAT, [1, 1, 4, 3]) # N x C x H x W
O = make_tensor_value_info('O', TensorProto.FLOAT, [1, 2, 3, 2]) # N x C x H x W

tensor_weight = from_array(np.array(
    [[[
        [-1, 1], [-1, 1]
        ]],

     [[
        [-1, -1], [1, 1]
         ]]], dtype=np.float32
    ), name="weight") # M x C x kH x kW
tensor_bias = from_array(np.array([0, 0], dtype=np.float32), name="bias")

node = make_node('Conv', ['I', 'weight', 'bias'], ['O'], kernel_shape=(2, 2), strides=(1, 1), pads=(0, 0, 0, 0))

graph = make_graph([node],
                   'lr',
                   [I],
                   [O],
                   [tensor_weight, tensor_bias]
                   )

onnx_model = make_model(graph)
check_model(onnx_model)
print(onnx_model)

with open("test.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

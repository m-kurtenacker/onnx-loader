import idx2numpy
import numpy as np
from scipy.signal import convolve2d as conv2d
import sys
import os
import subprocess

from onnx import TensorProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor, make_tensor_value_info)
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
    ), name="weight_1") # M x C x kH x kW
tensor_bias = from_array(np.array([0, 0], dtype=np.float32), name="bias_1")

constant_weight_np = np.array(
    [[[
        [-1, -1], [-1, -1]
        ]],

     [[
        [1, 1], [1, 1]
         ]]], dtype=np.float32
    ) # M x C x kH x kW
constant_bias_np = np.array([0.5, 0.5], dtype=np.float32)
constant_weight = make_node('Constant', [], ['weight_2'],
                            value=make_tensor(name='weight_2_const', data_type=NP_TYPE_TO_TENSOR_TYPE[constant_weight_np.dtype], dims=constant_weight_np.shape, vals=constant_weight_np))
constant_bias = make_node('Constant', [], ['bias_2'],
                            value=make_tensor(name='bias_2_const', data_type=NP_TYPE_TO_TENSOR_TYPE[constant_bias_np.dtype], dims=constant_bias_np.shape, vals=constant_bias_np))

node_1 = make_node('Conv', ['I', 'weight_1', 'bias_1'], ['C1'], kernel_shape=(2, 2), strides=(1, 1), pads=(0, 0, 0, 0))
node_2 = make_node('Conv', ['C1', 'weight_2', 'bias_2'], ['O'], kernel_shape=(2, 2), strides=(1, 1), pads=(0, 0, 0, 0))

graph = make_graph([constant_weight, constant_bias, node_1, node_2],
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

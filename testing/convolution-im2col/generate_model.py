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

shape = (64, 1, 224, 224)
out_channels = 16

I = make_tensor_value_info('I', TensorProto.FLOAT, shape) # N x C x H x W
O = make_tensor_value_info('O', TensorProto.FLOAT, [shape[0], out_channels, shape[2] - 1, shape[3] - 1]) # N x C x H x W

weight_array = np.array(np.random.random((out_channels, 1, 2, 2)), dtype=np.float32)
bias_array = np.array(np.random.random((out_channels,)), dtype=np.float32)

tensor_weight = from_array(weight_array, name="weight") # M x C x kH x kW
tensor_bias = from_array(bias_array, name="bias")

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

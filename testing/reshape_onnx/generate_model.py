# imports

from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model
from onnx.numpy_helper import from_array

import numpy as np

# inputs

# 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
A = make_tensor_value_info('A', TensorProto.FLOAT, [4, 3])

# outputs, the shape is left undefined

Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])

# nodes

reshape_1_size = from_array(np.array([2, 3, 2], dtype=np.int32), name='r1_size')
reshape_2_size = from_array(np.array([3, 4], dtype=np.int32), name='r2_size')

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32).reshape(2, 3, 2)
a = a / 256
a = from_array(a, name='a')

# It creates a node defined by the operator type MatMul,
# 'X', 'A' are the inputs of the node, 'XA' the output.
node1 = make_node('Reshape', ['A', 'r1_size'], ['R1'])
node2 = make_node('Add', ['R1', 'a'], ['A1'])
node3 = make_node('Reshape', ['A1', 'r2_size'], ['Y'])

# from nodes to graph
# the graph is built from the list of nodes, the list of inputs,
# the list of outputs and a name.

graph = make_graph([node1, node2, node3],  # nodes
                    'lr', # a name
                    [A],  # inputs
                    [Y],  # outputs
                    [reshape_1_size, reshape_2_size, a])  # initializers

# onnx graph
# there is no metadata in this case.

onnx_model = make_model(graph)

# Let's check the model is consistent,
# this function is described in section
# Checker and Shape Inference.
check_model(onnx_model)

# the work is done, let's display it...
print(onnx_model)

# The serialization
with open("test.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

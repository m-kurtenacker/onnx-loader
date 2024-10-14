import os
import onnx

from onnx.numpy_helper import to_array
from ConstantTensor import ConstantTensor

import sys

assert(len(sys.argv) >= 2)
ONNX_MODEL = sys.argv[1]
print("converting", ONNX_MODEL)
if len(sys.argv) >= 3:
    NETWORK_TOOLS_PATH = sys.argv[2]
else:
    NETWORK_TOOLS_PATH = "network_tools.thorin.json"

from pythorin import *

try:
    from IPython import embed
except ImportError:
    print("Importing IPython failed.")
    print("Install with ./venv/bin/pip install ipython")

model = onnx.load(ONNX_MODEL)
graph = model.graph

with Thorin("loader") as network:
    network.include(NETWORK_TOOLS_PATH)

    def load_initializer(onnx_node):
        data = to_array(onnx_node)
        name = onnx_node.name
        tensor = ConstantTensor(name, data)
        tensor.emit_thorin(tModule=network)
        tensor.emit_idx()

    def build_node(onnx_node):
        #Used to gather additional attributes that might be needed, e.g. padding
        if onnx_node.op_type == "Constant":
            data = to_array(onnx_node.attribute[0].t)
            name = onnx_node.output[0]
            tensor = ConstantTensor(name, data)
            tensor.emit_thorin(tModule=network)
            tensor.emit_idx()
        else:
            pass #This is deliberate, we do not want to emit code for stuff that is not a constant or initializer

    for initializer in graph.initializer:
        load_initializer(initializer)
    for node in graph.node:
        build_node(node)


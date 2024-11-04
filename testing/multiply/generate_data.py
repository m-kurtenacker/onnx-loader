import numpy as np

from ConstantTensor import ConstantTensor
from pythorin import *

import sys

assert(len(sys.argv) >= 1)
if len(sys.argv) >= 2:
    NETWORK_TOOLS_PATH = sys.argv[1]
else:
    NETWORK_TOOLS_PATH = "network_tools.thorin.json"

matrix_data = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [1, 1, 0],
                        [0, 0, 1]] , dtype=np.float32)
input_data = np.array([1, 2, 4] , dtype=np.float32)

tensor_matrix = ConstantTensor("M", matrix_data)
tensor_input = ConstantTensor("t", input_data)

with Thorin("loader") as loader:
    loader.include(NETWORK_TOOLS_PATH)

    tensor_matrix.emit_thorin(tModule=loader)
    tensor_input.emit_thorin(tModule=loader)

tensor_matrix.emit_idx()
tensor_input.emit_idx()

import numpy as np

from ConstantTensor import ConstantTensor
from pythorin import *

import sys

assert(len(sys.argv) >= 1)
if len(sys.argv) >= 2:
    NETWORK_TOOLS_PATH = sys.argv[1]
else:
    NETWORK_TOOLS_PATH = "network_tools.thorin.json"

rng = np.random.default_rng()

A = rng.random((8, 2), dtype=np.float32)
B = rng.random((2, 6), dtype=np.float32)
C = rng.random((8, 6), dtype=np.float32)

tensor_A = ConstantTensor("A", A)
tensor_B = ConstantTensor("B", B)
tensor_C = ConstantTensor("C", C)

with Thorin("loader") as loader:
    loader.include(NETWORK_TOOLS_PATH)

    tensor_A.emit_thorin(tModule=loader)
    tensor_B.emit_thorin(tModule=loader)
    tensor_C.emit_thorin(tModule=loader)

tensor_A.emit_idx()
tensor_B.emit_idx()
tensor_C.emit_idx()

from ConstantTensor import ConstantTensor
import numpy as np

ndarr = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]], dtype=np.uint8)

c = ConstantTensor("test", ndarr)

c.emit_idx()
c.emit_thorin()

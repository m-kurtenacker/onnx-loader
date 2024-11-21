import idx2numpy
import numpy as np

shape = (4, 4, 4, 4)

ndarr = np.random.random(shape) - 0.5
ndarr = np.array(ndarr, dtype=np.float32)

print(ndarr.shape)

idx2numpy.convert_to_file("test.idx", ndarr)

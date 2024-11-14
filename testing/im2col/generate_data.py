import idx2numpy
import numpy as np

shape = (1, 4, 3)

ndarr = np.random.random(shape) * 255
ndarr = np.array(ndarr, dtype=np.uint8)

print(ndarr.shape)

idx2numpy.convert_to_file("test.idx", ndarr)

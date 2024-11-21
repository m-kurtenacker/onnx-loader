import idx2numpy
import numpy as np

shape = (64, 1, 224, 224)

ndarr = np.random.random(shape)
ndarr = np.array(ndarr, dtype=np.float32)

print(ndarr.shape)

idx2numpy.convert_to_file("test.idx", ndarr)

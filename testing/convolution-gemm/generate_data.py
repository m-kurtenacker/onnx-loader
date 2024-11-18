import idx2numpy
import numpy as np

ndarr = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]], dtype=np.uint8)

print(ndarr.shape)

idx2numpy.convert_to_file("test.idx", ndarr)

import idx2numpy
import numpy as np

ndarr = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]], dtype=np.uint8)

print(ndarr.shape)

with open('test.idx', 'wb') as f_write:
    idx2numpy.convert_to_file(f_write, ndarr)

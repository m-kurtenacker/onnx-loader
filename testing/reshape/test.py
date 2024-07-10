import idx2numpy
import numpy as np
import sys
import os
import subprocess

if len(sys.argv) > 1:
    current_source_dir = os.path.dirname(sys.argv[0])
    current_binary_dir = sys.argv[1]
    subprocess.Popen([os.path.join(current_binary_dir, "test_reshape")]).communicate()

a = idx2numpy.convert_from_file("test.idx")
print(a)
onnx_result = idx2numpy.convert_from_file("result.idx")

#result = (((a[0] / 256) + (a[1] / 256)) * 256) % 256
intermediate = np.copy(a.reshape([1, 2, 3, 2]))
intermediate[0, 1, 1, 1] = 42
print(intermediate)
result = np.copy(intermediate.reshape([1, 3, 4]))

print(result)
print(onnx_result)

compare_distance = np.absolute(result - onnx_result)
print(compare_distance)

if np.max(compare_distance) > 0.01:
    sys.exit(1)
else:
    sys.exit(0)

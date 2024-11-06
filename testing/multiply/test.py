import idx2numpy
import numpy as np
import sys
import os
import subprocess

if len(sys.argv) > 1:
    current_source_dir = os.path.dirname(sys.argv[0])
    current_binary_dir = sys.argv[1]
    subprocess.Popen([os.path.join(current_binary_dir, "test_multiply")]).communicate()

M = idx2numpy.convert_from_file("M.idx")
t = idx2numpy.convert_from_file("t.idx")
print(M)
print(t)

onnx_result = idx2numpy.convert_from_file("result.idx")

result = np.dot(M, t)

print(result)
print(onnx_result)

compare_distance = np.absolute(result - onnx_result)
print(compare_distance)

if np.max(compare_distance) > 0.01:
    sys.exit(1)
else:
    sys.exit(0)

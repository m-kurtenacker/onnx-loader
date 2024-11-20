import idx2numpy
import numpy as np
import sys
import os
import subprocess

from onnx import load
from onnx.reference import ReferenceEvaluator
from onnx.numpy_helper import from_array

if len(sys.argv) > 1:
    current_source_dir = os.path.dirname(sys.argv[0])
    current_binary_dir = sys.argv[1]
    subprocess.Popen([os.path.join(current_binary_dir, "test_conv_im2col")]).communicate()

a = idx2numpy.convert_from_file("test.idx")
print(a)
print(a.shape)

with open("test.onnx", "rb") as f:
    model = load(f)

sess = ReferenceEvaluator(model)
a = np.array(a, dtype=np.float32)
feeds = {'I': a}
result = np.array(sess.run(None, feeds)[0])

print("np result")
print(result)
print(result.shape)
onnx_result = idx2numpy.convert_from_file("result.idx")
print("onnx result")
print(onnx_result)
print(onnx_result.shape)

print("compare")
compare_distance = np.absolute(result - onnx_result)
print(compare_distance)
print(np.max(compare_distance))

if np.max(compare_distance) > 0.01:
    sys.exit(1)
else:
    sys.exit(0)

import idx2numpy
import numpy as np
import sys
import os
import subprocess

if len(sys.argv) > 1:
    current_source_dir = os.path.dirname(sys.argv[0])
    current_binary_dir = sys.argv[1]
    subprocess.Popen([os.path.join(current_binary_dir, "test_gemm_256")]).communicate()

A = idx2numpy.convert_from_file("A.idx")
B = idx2numpy.convert_from_file("B.idx")
C = idx2numpy.convert_from_file("C.idx")

art_result = idx2numpy.convert_from_file("result.idx")

result = np.dot(A, B) + C;

print(result)
print(art_result)

compare_distance = np.absolute(result - art_result)
print(compare_distance)

if np.max(compare_distance) > 0.2:
    print(np.max(compare_distance))
    sys.exit(1)
else:
    sys.exit(0)

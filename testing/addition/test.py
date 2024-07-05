import idx2numpy

a = idx2numpy.convert_from_file("test.idx")
onnx_result = idx2numpy.convert_from_file("result.idx")

#print(a)
#print("")
#print(onnx_result)

result = (((a[0] / 256) + (a[1] / 256)) * 256) % 256

print(result)
print(onnx_result)
print(onnx_result - result)

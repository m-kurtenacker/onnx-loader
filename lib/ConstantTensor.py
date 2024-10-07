import idx2numpy
import numpy as np

from pythorin import *

import sys
sys.setrecursionlimit(5000)

assert(len(sys.argv) >= 1)
if len(sys.argv) >= 2:
    NETWORK_TOOLS_PATH = sys.argv[1]
else:
    NETWORK_TOOLS_PATH = "network_tools.thorin.json"

class ConstantTensor:
    def __init__(self, name, content):
        self.name = name
        self.content = content
        self.shape = self.content.shape

    def filename(self, ext="", pre=""):
        if ext == "":
            return pre + self.name
        else:
            return pre + self.name + "." + ext

    def emit_thorin(self):
        with Thorin(self.filename(pre="load_")) as module:
            module.include(NETWORK_TOOLS_PATH)

            build_tensor_thorin = module.find_imported_def("build_tensor_f32") #fn(mem, Buffer, i32, fn(mem, i32, fn(mem, i64)), fn(mem, Tensor))
            load_idx_thorin = module.find_imported_def("read_idx_int") #fn(mem, &[u8], fn(mem, Buffer))

            mem_type = build_tensor_thorin.type.args[0]
            buffer_type = build_tensor_thorin.type.args[1]
            i32_type = build_tensor_thorin.type.args[2]
            size_fn_type = build_tensor_thorin.type.args[3]
            i64_type = size_fn_type.args[2].args[1]
            tensor_type = build_tensor_thorin.type.args[4].args[-1]

            num_dimensions = ThorinConstant(i32_type, len(self.shape))
            thorin_dimensions = list(map(lambda x: ThorinConstant(i64_type, x), self.shape))
            sizes = ThorinDefiniteArray(i64_type, thorin_dimensions)

            with ThorinContinuation(size_fn_type) as (size_lambda, size_mem, dimension, size_return):
                r = ThorinExtract(sizes, dimension)
                size_lambda(size_return, size_mem, r)

            with ThorinContinuation(ThorinFnType([mem_type], tensor_type), internal="load_" + self.name, thorin=module) as (load_tensor, load_tensor_mem, ret_function):
                with ThorinContinuation(ThorinFnType([mem_type, buffer_type])) as (build_tensors, tensor_mem, input_buffer):
                    build_tensors(build_tensor_thorin, tensor_mem, input_buffer, num_dimensions, size_lambda, ret_function)

                load_tensor(load_idx_thorin, load_tensor_mem, thorinString(self.filename("idx")), build_tensors)

    def emit_idx(self):
        with open(self.filename("idx"), "wb") as f:
            idx2numpy.convert_to_file(f, self.content)

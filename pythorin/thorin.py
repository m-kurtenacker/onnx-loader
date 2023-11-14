import json
import ctypes
import os
import subprocess

from type_table import *
from irbuilder import *

class Thorin:
    def __init__(self, module_name):
        self.module = {"defs": [], "type_table": [], "module": module_name}
        self.module_name = module_name
        self.compiled = False

    #TODO: Use the thorin world for caching, don't cache information about the world inside defs.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.compile_module()

    def __del__(self):
        keep = os.environ.get("KEEP_BUILD_FILES")
        if (keep is None or keep == "0") and self.compiled:
            os.remove(self.module_name + ".thorin.json")
            os.remove(self.module_name + ".ll")
            os.remove(self.module_name + ".so")

    def add_def(self, thorin_def):
        assert(not self.compiled)

        return thorin_def.get(self.module)

    def compile(self):
        return json.dumps(self.module, indent=2)

    def compile_module(self):
        with open(self.module_name + ".thorin.json", "w+") as f:
            json.dump(self.module, f)

        subprocess.run(["anyopt", "--emit-llvm", "-o", self.module_name, self.module_name + ".thorin.json"])
        subprocess.run(["clang", "-shared", self.module_name + ".ll", "-o", self.module_name + ".so"])

        self.compiled = True

    def call_function(self, function_name, *args):
        assert(self.compiled)

        libc = ctypes.CDLL(self.module_name + ".so")

        return libc[function_name](*args)

    def __getattr__(self, function_name):
        return lambda *args : self.call_function(function_name, *args)

    def compile_function_jit(self, name, function, return_type, arg_types):
        mem_type = ThorinMemType()
        ret_fn_type = ThorinFnType([mem_type, return_type])
        fn_type = ThorinFnType([mem_type, *arg_types, ret_fn_type])

        with ThorinContinuation(fn_type, external=name, thorin=self) as (thorin_fn, mem_param, *param_list, ret_param):
            res = function(*param_list)

            thorin_fn(ret_param, mem_param, res)


        # Step 1: The function is "executed" with thorin values that track the operations in the function.
        #        This reqires some special precautions, for instance, range needs to be replaced. This is not a full fledged compiler, just a weird description system.
        # Step 2: Compile the final def to a python dict. Set some additional metadata, e.g. export the function using the correct name.
        # Step 3: Dump to json file.
        # Step 4: Compile the file using anyopt and clang. This should yield a shared object file.
        # Step 5: Link this file in and import the compiled function.

        # Design consideration: I would guess it to be a bad idea to build a thorin program that can deal with Python objects. Consequently, I need a translational layer. I would guess ctypes can be used to call pure C functions in .so files.?

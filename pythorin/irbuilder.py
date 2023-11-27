from .type_table import *

class ThorinDef:
    def __bool__(self):
        assert(False)
    def __init__(self):
        self.cache = ""
    def get(self, module):
        if self.cache == "":
            self.cache = self.compile(module)
        return self.cache
    def __add__(self, other):
        if isinstance(other, int):
            int_type = ThorinPrimType("qs32")
            other = ThorinConstant(int_type, other)
        return ThorinArithOp("add", [self, other])
    def __sub__(self, other):
        if isinstance(other, int):
            int_type = ThorinPrimType("qs32")
            other = ThorinConstant(int_type, other)
        return ThorinArithOp("sub", [self, other])
    def __mul__(self, other):
        if isinstance(other, int):
            int_type = ThorinPrimType("qs32")
            other = ThorinConstant(int_type, other)
        return ThorinArithOp("mul", [self, other])
    def __truediv__(self, other):
        if isinstance(other, int):
            int_type = ThorinPrimType("qs32")
            other = ThorinConstant(int_type, other)
        return ThorinArithOp("div", [self, other])
    def __lt__(self, other):
        if isinstance(other, int):
            int_type = ThorinPrimType("qs32")
            other = ThorinConstant(int_type, other)
        return ThorinCmp("lt", [self, other])
    def __le__(self, other):
        if isinstance(other, int):
            int_type = ThorinPrimType("qs32")
            other = ThorinConstant(int_type, other)
        return ThorinCmp("le", [self, other])
    def __gt__(self, other):
        if isinstance(other, int):
            int_type = ThorinPrimType("qs32")
            other = ThorinConstant(int_type, other)
        return ThorinCmp("gt", [self, other])
    def __ge__(self, other):
        if isinstance(other, int):
            int_type = ThorinPrimType("qs32")
            other = ThorinConstant(int_type, other)
        return ThorinCmp("ge", [self, other])

    def __rshift__(self, ptr):
        """This loads a value from memory. THIS IS NOT A SHIFT!"""
        return thorinLoadExtract(self, ptr)

    def __lshift__(self, store_ops):
        """This stores a value to memory. THIS IS NOT A SHIFT!"""
        ptr, value = store_ops
        return ThorinStore(self, ptr, value)


class ThorinArithOp(ThorinDef):
    def __init__(self, op, args):
        super().__init__()
        self.op = op
        self.args = args

    def compile(self, module):
        op = self.op
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_arithop_" + str(save_index)

        def_table.append({"type": "arithop", "name": name, "op": op, "args": args})
        return name


class ThorinMathOp(ThorinDef):
    def __init__(self, op, args):
        super().__init__()
        self.op = op
        self.args = args

    def compile(self, module):
        op = self.op
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_mathop_" + str(save_index)

        def_table.append({"type": "mathop", "name": name, "op": op, args: args})
        return name

class ThorinParameter(ThorinDef):
    def __init__(self, parent, index):
        super().__init__()
        self.parent = parent
        self.index = index

    def compile(self, module):
        name = self.parent.get(module) + "." + str(self.index)
        return name


class ThorinContinuation(ThorinDef):
    def __init__(self, type, external="", internal="", intrinsic="", app=None, filter=None, thorin=None):
        super().__init__()
        self.type = type
        self.external = external
        self.internal = internal
        self.intrinsic = intrinsic
        self.app = app
        self.filter = filter
        self.thorin = thorin

        self.parameters = []

        for i in range(0, len(type.args)):
            new_parameter = ThorinParameter(self, i)
            self.parameters.append(new_parameter)

    def __enter__(self):
        return (self, *self.parameters)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.thorin:
            self.thorin.add_def(self)

    def __call__(self, target, *args):
        self.app = (target, args)

    def compile(self, module):
        def_table = module["defs"]
        save_index = len(def_table)
        name = "_continuation_" + str(save_index)
        self.cache = name

        fntype = self.type.get(module)
        parameters = []
        for param in self.parameters:
            parameters.append(param.get(module))

        my_def = {"type": "continuation", "name": name, "fn_type": fntype, "arg_names": parameters}

        if self.external != "":
            my_def.update({"external": self.external})
        if self.internal != "":
            my_def.update({"internal": self.internal})
        if self.intrinsic != "":
            my_def.update({"intrinsic": self.intrinsic})

        def_table.append(my_def)

        my_filter = None
        if self.filter is not None:
            if self.filter == True:
                bool_type = ThorinPrimType("bool")
                true_const = ThorinConstant(bool_type, True)
                my_filter = ThorinFilter([true_const for i in range(0, len(self.parameters))]).get(module)
            elif self.filter == False:
                bool_type = ThorinPrimType("bool")
                false_const = ThorinConstant(bool_type, False)
                my_filter = ThorinFilter([false_const for i in range(0, len(self.parameters))]).get(module)
            else:
                my_filter = self.filter.get(module)

        if self.app:
            target, args = self.app
            compiled_target = target.get(module)
            compiled_args = []
            for arg in args:
                compiled_args.append(arg.get(module))
            app_def = {"type": "continuation", "name": name, "app": {"target": compiled_target, "args": compiled_args}}
            if my_filter:
                app_def.update({"filter": my_filter})
            def_table.append(app_def)

        return name


class ThorinConstant(ThorinDef):
    def __init__(self, type, value):
        super().__init__()
        self.type = type
        self.value = value

    def compile(self, module):
        const_type = self.type.get(module)
        value = self.value

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_constant_" + str(save_index)

        def_table.append({"type": "const", "name": name, "const_type": const_type, "value": value})
        return name


class ThorinTop(ThorinDef):
    def __init__(self, type):
        super().__init__()
        self.type = type
    def compile(self, module):
        const_type = self.type.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_top_" + str(save_index)

        def_table.append({"type": "top", "name": name, "const_type": const_type})
        return name


class ThorinBottom(ThorinDef):
    def __init__(self, type):
        super().__init__()
        self.type = type

    def compile(self, module):
        const_type = self.type.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_bottom_" + str(save_index)

        def_table.append({"type": "bottom", "name": name, "const_type": const_type})

        return name


class ThorinCmp(ThorinDef):
    def __init__(self, op, args):
        super().__init__()
        self.op = op
        self.args = args

    def compile(self, module):
        op = self.op
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_cmp_" + str(save_index)

        def_table.append({"type": "cmp", "name": name, "op": op, "args": args})
        return name


class ThorinLEA(ThorinDef):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def compile(self, module):
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_lea_" + str(save_index)

        def_table.append({"type": "lea", "name": name, "args": args})
        return name


class ThorinLoad(ThorinDef):
    def __init__(self, mem, pointer):
        super().__init__()
        self.mem = mem
        self.pointer = pointer

    def compile(self, module):
        mem = self.mem.get(module)
        pointer = self.pointer.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_load_" + str(save_index)

        def_table.append({"type": "load", "name": name, "args": [mem, pointer]})
        return name


class ThorinExtract(ThorinDef):
    def __init__(self, aggregate, index):
        super().__init__()
        self.aggregate = aggregate
        if isinstance(index, ThorinDef):
            self.index = index
        elif isinstance(index, int):
            unsigned_type = ThorinPrimType("qu32")
            self.index = ThorinConstant(unsigned_type, index)
        else:
            raise Exception("Not supported")

    def compile(self, module):
        aggregate = self.aggregate.get(module)
        index = self.index.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_extract_" + str(save_index)

        def_table.append({"type": "extract", "name": name, "args": [aggregate, index]})
        return name


class ThorinInsert(ThorinDef):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def compile(self, module):
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_insert_" + str(save_index)

        def_table.append({"type": "insert", "name": name, "args": args})
        return name


class ThorinCast(ThorinDef):
    def __init__(self, source, type):
        super().__init__()
        self.source = source
        self.type = type

    def compile(self, module):
        source = self.source.get(module)
        type = self.type.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_cast_" + str(save_index)

        def_table.append({"type": "cast", "name": name, "source": source, "target_type": type})
        return name


class ThorinBitcast(ThorinDef):
    def __init__(self, source, type):
        super().__init__()
        self.source = source
        self.type = type

    def compile(self, module):
        source = self.source.get(module)
        type = self.type.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_bitcast_" + str(save_index)

        def_table.append({"type": "bitcast", "name": name, "source": source, "target_type": type})
        return name


class ThorinRun(ThorinDef):
    def __init__(self):
        super().__init__()
    def compile(self, module):
        def_table = module["defs"]
        save_index = len(def_table)
        name = "_run_" + str(save_index)
        def_table.append({"type": "run", "name": name})
        return name


class ThorinHlt(ThorinDef):
    def __init__(self, target):
        super().__init__()
        self.target = target

    def compile(self, module):
        target = self.target.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_hlt_" + str(save_index)

        def_table.append({"type": "hlt", "name": name, "target": target})
        return name


class ThorinStore(ThorinDef):
    def __init__(self, mem, pointer, value):
        super().__init__()
        self.mem = mem
        self.pointer = pointer
        if isinstance(value, ThorinDef):
            self.value = value
        elif isinstance(value, int):
            int_type = ThorinPrimType("qs32")
            self.value = ThorinConstant(int_type, value)
        else:
            raise Exception("Not supported")

    def compile(self, module):
        mem = self.mem.get(module)
        pointer = self.pointer.get(module)
        value = self.value.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_store_" + str(save_index)

        def_table.append({"type": "store", "name": name, "args": [mem, pointer, value]})
        return name


class ThorinEnter(ThorinDef):
    def __init__(self, mem):
        super().__init__()
        self.mem = mem

    def compile(self, module):
        mem = self.mem.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_enter_" + str(save_index)

        def_table.append({"type": "enter", "name": name, "mem": mem})
        return name


class ThorinSlot(ThorinDef):
    def __init__(self, frame, type):
        super().__init__()
        self.frame = frame
        self.type = type

    def compile(self, module):
        type = self.type.get(module)
        frame = self.frame.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_slot_" + str(save_index)

        def_table.append({"type": "slot", "name": name, "frame": frame, "target_type": type})
        return name


class ThorinDefiniteArray(ThorinDef):
    def __init__(self, elem_type, args):
        super().__init__()
        self.elem_type = elem_type
        self.args = args

    def compile(self, module):
        elem_type = self.elem_type.get(module)
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_definitearray_" + str(save_index)

        def_table.append({"type": "def_array", "name": name, "elem_type": elem_type, "args": args})
        return name


class ThorinIndefiniteArray(ThorinDef):
    def __init__(self, elem_type, dim):
        super().__init__()
        self.elem_type = elem_type
        self.dim = dim

    def compile(self, module):
        elem_type = self.elem_type.get(module)
        dim = self.dim.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_indefinitearray_" + str(save_index)

        def_table.append({"type": "indef_array", "name": name, "elem_type": elem_type, "dim": dim})
        return name


class ThorinGlobal(ThorinDef):
    def __init__(self, init, mutable=False, external=None):
        super().__init__()
        self.init = init
        self.external = external
        self.mutable = mutable

    def compile(self, module):
        init = self.init.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_global_" + str(save_index)

        my_def = {"type": "global", "name": name, "mutable": self.mutable, "init": init}

        if self.external is not None:
            my_def.update({"external": self.external})

        def_table.append(my_def)
        return name


class ThorinClosure(ThorinDef):
    def __init__(self, args, closure_type):
        super().__init__()
        self.args = args
        self.closure_type = closure_type

    def compile(self, module):
        closure_type = self.closure_type.get(module)
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_closure_" + str(save_index)

        def_table.append({"type": "closure", "name": name, "closure_type": closure_type, "args": args})
        return name


class ThorinStruct(ThorinDef):
    def __init__(self, args, struct_type):
        super().__init__()
        self.args = args
        self.struct_type = struct_type

    def compile(self, module):
        struct_type = self.struct_type.get(module)
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_struct_" + str(save_index)

        def_table.append({"type": "struct", "name": name, "struct_type": struct_type, "args": args})
        return name


class ThorinTuple(ThorinDef):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def compile(self, module):
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_tuple_" + str(save_index)

        def_table.append({"type": "tuple", "name": name, "args": args})
        return name


class ThorinVector(ThorinDef):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def compile(self, module):
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_vector_" + str(save_index)

        def_table.append({"type": "vector", "name": name, "args": args})
        return name


class ThorinAlloc(ThorinDef):
    def __init__(self, target_type, args):
        super().__init__()
        self.target_type = target_type
        self.args = args

    def compile(self, module):
        target_type = self.target_type.get(module)
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_alloc_" + str(save_index)

        def_table.append({"type": "alloc", "name": name, "target_type": target_type, "args": args})
        return name


class ThorinKnown(ThorinDef):
    def __init__(self, int_def):
        super().__init__()
        self.int_def = int_def

    def compile(self, module):
        int_def = self.int_def.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_known_" + str(save_index)

        def_table.append({"type": "known", "name": name, "def": int_def})
        return name


class ThorinSizeof(ThorinDef):
    def __init__(self, target_type):
        super().__init__()
        self.target_type = target_type

    def compile(self, module):
        target_type = self.target_type.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_sizeof_" + str(save_index)

        def_table.append({"type": "sizeof", "name": name, "target_type": target_type})
        return name


class ThorinAlignof(ThorinDef):
    def __init__(self, target_type):
        super().__init__()
        self.target_type = target_type

    def compile(self, module):
        target_type = self.target_type.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_alignof_" + str(save_index)

        def_table.append({"type": "alignof", "name": name, "target_type": target_type})
        return name


class ThorinSelect(ThorinDef):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def compile(self, module):
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_select_" + str(save_index)

        def_table.append({"type": "select", "name": name, "args": args})
        return name


class ThorinFilter(ThorinDef):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def compile(self, module):
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_filter_" + str(save_index)

        def_table.append({"type": "filter", "name": name, "args": args})
        return name


class ThorinVariant(ThorinDef):
    def __init__(self, variant_type, value, index):
        super().__init__()
        self.variant_type = variant_type
        self.value = value
        self.index = index

    def compile(self, module):
        variant_type = self.variant_type.get(module)
        value = self.value.get(module)
        index = self.index

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_variant_" + str(save_index)

        def_table.append({"type": "variant", "name": name, "variant_type": variant_type, "value": value, "index": index})
        return name


class ThorinVariantExtract(ThorinDef):
    def __init__(self, value, index):
        super().__init__()
        self.value = value
        self.index = index

    def compile(self, module):
        value = self.value.get(module)
        index = self.index

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_variantextract_" + str(save_index)

        def_table.append({"type": "variantextract", "name": name, "value": value, "index": index})
        return name


class ThorinVariantIndex(ThorinDef):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def compile(self, module):
        value = self.value.get(module)

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_variantindex_" + str(save_index)

        def_table.append({"type": "variantindex", "name": name, "value": value})
        return name


class ThorinAssembly(ThorinDef):
    def __init__(self, asm_type, inputs, asm_template, in_constraints, out_constraints, clobbers):
        super().__init__()
        self.asm_type = asm_type
        self.inputs = inputs
        self.asm_template = asm_template
        self.in_constraints = in_constraints
        self.out_constraints = out_constraints
        self.clobbers = clobbers

    def compile(self, module):
        asm_type = self.asm_type.get(module)
        inputs = []
        for i in self.inputs:
            inputs.append(i.get(module))

        def_table = module["defs"]
        save_index = len(def_table)
        name = "_assembly_" + str(save_index)

        def_table.append({"type": "assembly", "name": name, "asm_type": asm_type, "inputs": inputs, "asm_template": self.asm_template, "input_constraints": self.in_constraints, "output_constraints": self.out_constraints, "clobbers": self.clobbers})
        return name

#Helper functions that construct common complex patterns

def thorinLoadExtract(mem, ptr):
    load = ThorinLoad(mem, ptr)
    return (ThorinExtract(load, 0), ThorinExtract(load, 1))

def thorinEnterExtract(mem):
    res = ThorinEnter(mem)
    return (ThorinExtract(res, 0), ThorinExtract(res, 1))

def thorinBranch(target, mem_param, cond_param, branch_true=None, branch_false=None):
    if branch_true is None or branch_false is None:
        mem_type = ThorinMemType()
        mem_fn_type = ThorinFnType([mem_type])

    if branch_true is None:
        branch_true = ThorinContinuation(mem_fn_type)
    if branch_false is None:
        branch_false = ThorinContinuation(mem_fn_type)

    bool_type = ThorinPrimType("bool")
    branch_type = ThorinFnType([mem_type, bool_type, mem_fn_type, mem_fn_type])
    branch_int = ThorinContinuation(branch_type, intrinsic="branch")
    target(branch_int, mem_param, cond_param, branch_true, branch_false)

    return branch_true, branch_true.parameters[0], branch_false, branch_false.parameters[0]

def thorinBranchFn(mem_param, cond_param, branch_true=None, branch_false=None):
    mem_type = ThorinMemType()
    mem_fn_type = ThorinFnType([mem_type])

    with ThorinContinuation(mem_fn_type) as (branch_true_block, branch_true_mem):
        if branch_true is not None:
            branch_true(branch_true_block, branch_true_mem)

    with ThorinContinuation(mem_fn_type) as (branch_false_block, branch_false_mem):
        if branch_false is not None:
            branch_false(branch_false_block, branch_false_mem)

    bool_type = ThorinPrimType("bool")
    branch_type = ThorinFnType([mem_type, bool_type, mem_fn_type, mem_fn_type])
    branch_int = ThorinContinuation(branch_type, intrinsic="branch")

    return (branch_int, mem_param, cond_param, branch_true_block, branch_false_block)

def thorinString(content):
    strlen = len(content)
    bytestr = list(content.encode("utf-8")) + [0]

    u8type = ThorinPrimType("pu8")

    def_array = ThorinDefiniteArray(u8type, map(lambda x: ThorinConstant(u8type, x), bytestr))
    global_array = ThorinGlobal(def_array)
    bitcast_array = ThorinBitcast(global_array, ThorinPointerType(ThorinIndefiniteArrayType(u8type)))

    return bitcast_array

def thorinRangeFn(mem_param, lower, upper, step, body_fn, return_fn):
    int_type = ThorinPrimType("qs32")
    mem_type = ThorinMemType()

    ret_fn_type = ThorinFnType([mem_type, int_type])
    export_fn_type = ThorinFnType([mem_type, int_type, int_type, ret_fn_type])
    mem_fn_type = ThorinFnType([mem_type])
    body_fn_type = ThorinFnType([mem_type, int_type, mem_fn_type])
    range_fn_type = ThorinFnType([mem_type, int_type, int_type])

    with ThorinContinuation(body_fn_type) as (body_block, body_mem, i, next_fn):
        body_fn(body_block, body_mem, i, next_fn)

    with ThorinContinuation(mem_fn_type) as (return_block, return_mem):
        return_fn(return_block, return_mem)

    with ThorinContinuation(range_fn_type) as (range_fn, range_mem, lower_param, upper_param):
        def branch_true(branch_true, true_mem): #Loop Body
            with ThorinContinuation(mem_fn_type) as (continue_fn, continue_mem):
                next_lower = lower_param + step
                continue_fn(range_fn, continue_mem, next_lower, upper_param)

            branch_true(body_block, true_mem, lower_param, continue_fn)

        def branch_false(branch_false, false_mem): #Loop Exit
            branch_false(return_block, false_mem)

        range_fn(*thorinBranchFn(range_mem, lower_param < upper_param, branch_true, branch_false))

    return (range_fn, mem_param, lower, upper)

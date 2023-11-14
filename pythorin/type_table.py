class ThorinType:
    def __init__(self):
        self.cache = ""
    def get(self, module):
        if self.cache == "":
            self.cache = self.compile(module)
        return self.cache
    def compile(self, module):
        raise Exception("Not implemented")


class ThorinDefiniteArrayType(ThorinType):
    def __init__(self, target_type, length):
        super().__init__()
        self.target_type = target_type
        self.length = length

    def compile(self, module):
        target_type = self.target_type.get(module)

        type_table = module["type_table"]
        save_index = len(type_table)
        name = "_def_array_" + str(save_index)

        type_table.append({"type": "def_array", "name": name, "lenght": self.length, "args": [target_type]})
        return name


class ThorinIndefiniteArrayType(ThorinType):
    def __init__(self, target_type):
        super().__init__()
        self.target_type = target_type

    def compile(self, module):
        target_type = self.target_type.get(module)

        type_table = module["type_table"]
        save_index = len(type_table)
        name = "_indef_array_" + str(save_index)

        type_table.append({"type": "indef_array", "name": name, "args": [target_type]})
        return name


class ThorinBottomType(ThorinType):
    def __init__(self):
        super().__init__()

    def compile(self, module):
        type_table = module["type_table"]
        save_index = len(type_table)
        name = "_bottom_" + str(save_index)

        type_table.append({"type": "bottom", "name": name})
        return name


class ThorinFnType(ThorinType):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def compile(self, module):
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        type_table = module["type_table"]
        save_index = len(type_table)
        name = "_fn_" + str(save_index)

        type_table.append({"type": "function", "name": name, "args": args})
        return name


class ThorinClosureType(ThorinType):
    def __init__(self):
        super().__init__()
    def compile(self, module):
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        type_table = module["type_table"]
        save_index = len(type_table)
        name = "_closure_" + str(save_index)

        type_table.append({"type": "closure", "name": name, "args": args})
        return name


class ThorinFrameType(ThorinType):
    def __init__(self):
        super().__init__()

    def compile(self, module):
        type_table = module["type_table"]
        save_index = len(type_table)
        name = "_frame_" + str(save_index)

        type_table.append({"type": "frame", "name": name})
        return name


class ThorinMemType(ThorinType):
    def __init__(self):
        super().__init__()

    def compile(self, module):
        type_table = module["type_table"]
        save_index = len(type_table)
        name = "_mem_" + str(save_index)

        type_table.append({"type": "mem", "name": name})
        return name


class ThorinStructType(ThorinType):
    def __init__(self, struct_name, formated_args=None):
        super().__init__()
        self.struct_name = struct_name
        self.formated_args = formated_args

    def compile(self, module):
        assert(self.formated_args)

        type_table = module["type_table"]
        save_index = len(type_table)
        name = "_struct_" + str(save_index)
        self.cache = name

        arg_names = []
        for name, arg in formated_args:
            arg_names.append(name)

        type_table.append({"type": "struct", "name": name, "struct_name": self.struct_name, "arg_names": arg_names})

        args = []
        for name, arg in formated_args:
            args.append(arg.get(module))

        type_table.append({"type": "struct", "name": name, "arg_names": arg_names, "args": args})
        return name


class ThorinVariantType(ThorinType):
    def __init__(self, variant_name, formated_args=None):
        super().__init__()
        self.variant_name = variant_name
        self.formated_args = formated_args

    def compile(self, module):
        assert(self.formated_args)

        type_table = module["type_table"]
        save_index = len(type_table)
        name = "_variant_" + str(save_index)
        self.cache = name

        arg_names = []
        for name, arg in formated_args:
            arg_names.append(name)

        type_table.append({"type": "variant", "name": name, "variant_name": self.variant_name, "arg_names": arg_names})

        args = []
        for name, arg in formated_args:
            args.append(arg.get(module))

        type_table.append({"type": "variant", "name": name, "arg_names": arg_names, "args": args})
        return name


class ThorinTupleType(ThorinType):
    def __init__(self, args):
        super().__init__()
        self.args = args
    def compile(self, module):
        args = []
        for arg in self.args:
            args.append(arg.get(module))

        type_table = module["type_table"]
        save_index = len(type_table)
        name = "_tuple_" + str(save_index)

        type_table.append({"type": "tuple", "name": name, "args": args})
        return name


class ThorinPrimType(ThorinType):
    def __init__(self, tag, length):
        super().__init__()
        self.tag = tag
        self.length = length

    def compile(self, module):
        tag = self.tag
        length = self.length

        type_table = module["type_table"]
        save_index = len(type_table)
        name = "_prim_" + str(save_index)

        type_table.append({"type": "prim", "name": name, "tag": tag, "length": length})
        return name


class ThorinPointerType(ThorinType):
    def __init__(self, pointee, length, device=None, addrspace=None):
        super().__init__()
        self.pointee = pointee
        self.length = length
        self.device = device
        self.addrspace = addrspace

    def compile(self, module):
        pointee = self.pointee.get(module)
        length = self.length

        type_table = module["type_table"]
        save_index = len(type_table)
        name = "_ptr_" + str(save_index)

        my_def = {"type": "ptr", "name": name, "length": length, "args": [pointee]}
        if self.device:
            my_def.update({"device": device})
        if self.addrspace:
            my_def.update({"addrspace": addrspace})

        type_table.append(my_def)
        return name

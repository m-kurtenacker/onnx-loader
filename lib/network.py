import onnx

import sys

assert(len(sys.argv) >= 2)
ONNX_MODEL = sys.argv[1]
print("converting", ONNX_MODEL)
if len(sys.argv) >= 3:
    NETWORK_TOOLS_PATH = sys.argv[2]
else:
    NETWORK_TOOLS_PATH = "network_tools.thorin.json"

from pythorin import *

try:
    from IPython import embed
except ImportError:
    print("Importing IPython failed.")
    print("Install with ./venv/bin/pip install ipython")

import sys
sys.setrecursionlimit(5000)


model = onnx.load(ONNX_MODEL)
graph = model.graph
infered_model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)


#Setup all types that are required in the network execution stack
mem_type = ThorinMemType()
f32_type = ThorinPrimType("qf32")
i32_type = ThorinPrimType("qs32")
i64_type = ThorinPrimType("qs64")
i8_type = ThorinPrimType("qs8")

i64_arr_type = ThorinPointerType(ThorinIndefiniteArrayType(i64_type))

return_type = ThorinFnType([mem_type])
exec_type = ThorinFnType([mem_type], ThorinFnType([mem_type]))

thorin_constant_producer_type = ThorinFnType([mem_type, ThorinTupleType([i32_type, i64_arr_type])])


def alloc_initializer(entry_mem, finish_cont, dimensions):
    #print("Alloc initializer with", dimensions)
    thorin_dimensions = list(map(lambda x: ThorinConstant(i64_type, x), dimensions))
    sizes = ThorinDefiniteArray(i64_type, thorin_dimensions)

    local_tensor_type = finish_cont.type.args[1]

    with ThorinContinuation(local_tensor_type.formated_args[3][1], filter=True) as (size_lambda, size_mem, dimension, size_return):
        r = ThorinExtract(sizes, dimension)
        size_lambda(size_return, size_mem, r)

    if local_tensor_type == tensor_i64_type:
        alloc_function = alloc_initializer_i64_thorin
    else:
        alloc_function = alloc_initializer_f32_thorin

    return (alloc_function, entry_mem, ThorinConstant(i32_type, len(dimensions)), size_lambda, finish_cont)


def alloc_and_load_tensor(entry_mem, finish_cont, dimensions, matrix_name):
    model_name = thorinString(ONNX_MODEL)
    matrix_name = thorinString(matrix_name)

    return_fn_type = finish_cont.type
    local_tensor_type = finish_cont.type.args[1]

    with ThorinContinuation(return_fn_type) as (alloc_continue, alloc_mem, tensor):
        with ThorinContinuation(return_type) as (load_cont, finish_mem):
            load_cont(finish_cont, finish_mem, tensor)
        with ThorinContinuation(exec_type, filter=True) as (load_return, load_mem, load_int):
            load_return(load_int, load_mem, load_cont);

        if local_tensor_type == tensor_i64_type:
            load_matrix_into = load_matrix_into_i64
        else:
            load_matrix_into = load_matrix_into_f32

        alloc_continue(load_matrix_into, alloc_mem, tensor, model_name, matrix_name, load_return)

    return alloc_initializer(entry_mem, alloc_continue, dimensions)
    #return alloc_initializer(entry_mem, finish_cont, dimensions) #Do not load stuf, only allocation.


def load_initializer(onnx_node):
    dimensions = onnx_node.dims
    data_type = onnx_node.data_type

    if data_type == 7:
        local_tensor_return_type = tensor_i64_return_type
    else:
        local_tensor_return_type = tensor_return_type

    with ThorinContinuation(local_tensor_return_type) as (return_cont, return_mem, result_tensor):
        #dimensions.reverse()
        call_function = lambda in_cont, in_mem: in_cont(*alloc_and_load_tensor(in_mem, return_cont, dimensions, onnx_node.name))
        return_function = lambda out_cont, *out_param: return_cont(out_cont, return_mem, *out_param)

        update_node = {"result": result_tensor, "call": call_function, "cont": return_function, "block": (return_cont, return_mem), "onnx_node": onnx_node}
        nodes.update({onnx_node.name : update_node})
        return onnx_node.name


def convert_to_global_array(input_array, thorin_type):
    thorin_array = ThorinDefiniteArray(thorin_type, [ThorinConstant(thorin_type, x) for x in input_array])
    glob = ThorinGlobal(thorin_array)
    glob_cast = ThorinBitcast(glob, ThorinPointerType(ThorinIndefiniteArrayType(thorin_type)))
    num_dims = ThorinConstant(i32_type, len(input_array))
    return ThorinTuple([num_dims, glob_cast])


def translate_operation(onnx_node):
    #The functions referenced here are loaded from the network later. Python is awsome sometimes.
    if onnx_node.op_type == "Gemm" or "Linear" in onnx_node.op_type:
        return (mat_gemm_impl, mat_gemm_setup)
    elif onnx_node.op_type == "Flatten" or "Flatten" in onnx_node.op_type:
        return (mat_flatten, None)
    elif onnx_node.op_type == "Add" or "Add" in onnx_node.op_type:
        return (mat_add_impl, mat_add_setup)
    elif onnx_node.op_type == "Reshape" or "_view" in onnx_node.op_type:
        input_size = nodes[onnx_node.input[1]]["onnx_node"]
        if hasattr(input_size, "op_type") and input_size.op_type == "Constant":
            return (mat_reshape_const, None)
        else:
            return (mat_reshape_impl, mat_reshape_setup)
    elif onnx_node.op_type == "Relu" or "relu" in onnx_node.op_type:
        return (mat_relu, None)
    elif onnx_node.op_type == "LRN" or "LocalResponseNorm" in onnx_node.op_type:
        return (mat_lrn, None)
    elif onnx_node.op_type == "LogSoftmax" or "log_softmax" in onnx_node.op_type:
        return (mat_log_softmax, None)
    elif onnx_node.op_type == "Softmax" or "softmax" in onnx_node.op_type:
        return (mat_softmax, None)
    elif onnx_node.op_type == "Conv" or "conv" in onnx_node.op_type:
        print("This should hit a special case in build_node.")
        assert(False)
    elif onnx_node.op_type == "MaxPool" or "max_pool" in onnx_node.op_type:
        return (mat_max_pool, None)
    elif onnx_node.op_type == "AveragePool" or "avg_pool" in onnx_node.op_type:
        return (mat_avg_pool, None)
    elif onnx_node.op_type == "Dropout" or "dropout" in onnx_node.op_type:
        return (mat_dropout, None)
    elif onnx_node.op_type == "Concat" or "SequenceConstruct" in onnx_node.op_type:
        return (mat_concat, None)
    elif "aten_cat" in onnx_node.op_type:
        return (mat_dropout, None)
    else:
        print("op unknown:", onnx_node.op_type, "at", onnx_node.name)
        print(onnx_node)
        assert(False)


def build_node(onnx_node):
    #Used to gather additional attributes that might be needed, e.g. padding
    def get_attributes(*args):
        results = [None for _ in range(0, len(args))]
        for attribute in onnx_node.attribute:
            for index, arg in zip(range(0, len(args)), args):
                if attribute.name == arg:
                    results[index] = attribute.ints
        return results

    #Gather the output shape for all nodes.
    output_shape = []
    if onnx_node.output[0] == output_name:
        start_index = 0
        dim_vector = graph.output[0].type.tensor_type.shape.dim
        if dim_vector[0].dim_param == "batch_size":
            starting_index = 1
        output_shape = [x.dim_value for x in dim_vector[start_index:]]
    else:
        for info in infered_model.graph.value_info:
            if info.name == onnx_node.output[0]:
                starting_index = 0
                dim_vector = info.type.tensor_type.shape.dim
                if dim_vector[0].dim_param == "batch_size":
                    starting_index = 1
                output_shape = [x.dim_value for x in dim_vector[starting_index:]]

                output_shape.reverse()
                break
    if output_shape == []:
        print("Node has no output shape?:", onnx_node)
    assert(output_shape != [])

    if onnx_node.op_type == "Constant":
        if onnx_node.attribute[0].t.data_type == 6: #i32
            const_type = i32_type
        elif onnx_node.attribute[0].t.data_type == 7: #i64
            const_type = i64_type
        else:
            assert(False)

        num_dims = len(onnx_node.attribute[0].t.dims)

        constants = list(onnx.numpy_helper.to_array(onnx_node.attribute[0].t)[1:])
        thorin_constants = list(map(lambda x: ThorinConstant(const_type, int(x)), constants))
        thorin_def_array = ThorinDefiniteArray(const_type, thorin_constants)

        with ThorinContinuation(thorin_constant_producer_type) as (return_cont, return_mem, result_tuple):
            addr_ptr = ThorinSlot(frame, ThorinDefiniteArrayType(i64_type, num_dims))
            return_mem = return_mem << (addr_ptr, thorin_def_array)

            addr_ptr_opaque = ThorinBitcast(addr_ptr, i64_arr_type)
            shape_tuple = ThorinTuple([ThorinConstant(i32_type, num_dims), addr_ptr_opaque]);

            call_function = lambda in_cont, in_mem: in_cont(return_cont, in_mem, shape_tuple)
            return_function = lambda out_cont, *out_param: return_cont(out_cont, return_mem, *out_param)

            update_node = {"result": result_tuple, "call": call_function, "cont": return_function, "block": (return_cont, return_mem), "onnx_node": onnx_node}
            nodes.update({onnx_node.output[0] : update_node})

        return onnx_node.output[0], None

    elif onnx_node.op_type == "Transpose":
        perm, = get_attributes("perm")

        perm_global = convert_to_global_array(perm, i64_type)
        output_shape_global = convert_to_global_array(output_shape, i64_type)

        attributes = [perm_global, output_shape_global]

        thorin_operation = mat_transpose
        return_fn_type = thorin_operation.type.args[-1]

        with ThorinContinuation(return_fn_type) as (return_cont, return_mem, result_tensor):
            call_function = lambda in_cont, in_mem: in_cont(thorin_operation, in_mem, *[nodes[name]["result"] for name in onnx_node.input], *attributes, return_cont)
            return_function = lambda out_cont, *out_param: return_cont(out_cont, return_mem, *out_param)

            update_node = {"result": result_tensor, "call": call_function, "cont": return_function, "block": (return_cont, return_mem), "onnx_node": onnx_node}
            nodes.update({onnx_node.output[0] : update_node})

        return onnx_node.output[0], None

    elif onnx_node.op_type == "Conv" or "conv" in onnx_node.op_type or onnx_node.op_type == "MaxPool" or "max_pool" in onnx_node.op_type or onnx_node.op_type == "AveragePool":
        shape, strides, padding = get_attributes("kernel_shape", "strides", "pads")

        shape_global = convert_to_global_array(shape, i64_type)
        stride_global = convert_to_global_array(strides, i64_type)
        padding_global = convert_to_global_array(padding, i64_type)
        output_shape_global = convert_to_global_array(output_shape, i64_type)

        attributes = [output_shape_global, shape_global, stride_global, padding_global]

        with ThorinContinuation(mat_conv_setup.type.args[-1]) as (setup_cont, setup_mem, setup_result):
            call_function = lambda in_cont, in_mem: in_cont(mat_conv_setup, in_mem, *attributes, setup_cont)
            return_function = lambda out_cont, *out_param: setup_cont(out_cont, setup_mem, *out_param)

            update_node = {"result": setup_result, "call": call_function, "cont": return_function, "block": (setup_cont, setup_mem), "onnx_node": onnx_node}
            nodes.update({onnx_node.output[0] + "_setup" : update_node})

        with ThorinContinuation(mat_conv_exec.type.args[-1]) as (exec_cont, exec_mem, result_tensor):
            call_function = lambda in_cont, in_mem: in_cont(mat_conv_exec, in_mem, *[nodes[name]["result"] for name in onnx_node.input], *attributes, setup_result, exec_cont)
            return_function = lambda out_cont, *out_param: exec_cont(out_cont, exec_mem, *out_param)

            update_node = {"result": result_tensor, "call": call_function, "cont": return_function, "block": (exec_cont, exec_mem), "onnx_node": onnx_node}
            nodes.update({onnx_node.output[0] : update_node})

        return onnx_node.output[0], (onnx_node.output[0] + "_setup")

    else:
        (thorin_operation, setup_operation) = translate_operation(onnx_node)

        output_shape_global = convert_to_global_array(output_shape, i64_type)

        attributes = [output_shape_global]

        if setup_operation is not None:
            with ThorinContinuation(setup_operation.type.args[-1]) as (setup_cont, setup_mem, setup_result):
                call_function = lambda in_cont, in_mem: in_cont(setup_operation, in_mem, *attributes, setup_cont)
                return_function = lambda out_cont, *out_param: setup_cont(out_cont, setup_mem, *out_param)

                update_node = {"result": setup_result, "call": call_function, "cont": return_function, "block": (setup_cont, setup_mem), "onnx_node": onnx_node}
                nodes.update({onnx_node.output[0] + "_setup" : update_node})

            with ThorinContinuation(thorin_operation.type.args[-1]) as (exec_cont, exec_mem, result_tensor):
                call_function = lambda in_cont, in_mem: in_cont(thorin_operation, in_mem, *[nodes[name]["result"] for name in onnx_node.input], *attributes, setup_result, exec_cont)
                return_function = lambda out_cont, *out_param: exec_cont(out_cont, exec_mem, *out_param)

                update_node = {"result": result_tensor, "call": call_function, "cont": return_function, "block": (exec_cont, exec_mem), "onnx_node": onnx_node}
                nodes.update({onnx_node.output[0] : update_node})

            return onnx_node.output[0], (onnx_node.output[0] + "_setup")

        else:
            with ThorinContinuation(thorin_operation.type.args[-1]) as (return_cont, return_mem, result_tensor):
                call_function = lambda in_cont, in_mem: in_cont(thorin_operation, in_mem, *[nodes[name]["result"] for name in onnx_node.input], *attributes, return_cont)
                return_function = lambda out_cont, *out_param: return_cont(out_cont, return_mem, *out_param)

                update_node = {"result": result_tensor, "call": call_function, "cont": return_function, "block": (return_cont, return_mem), "onnx_node": onnx_node}
                nodes.update({onnx_node.output[0] : update_node})

            return onnx_node.output[0], None


with Thorin("network") as network:
    network.include(NETWORK_TOOLS_PATH)

    alloc_initializer_f32_thorin = network.find_imported_def("alloc_tensor_f32")
    alloc_initializer_i64_thorin = network.find_imported_def("alloc_tensor_i64")
    load_matrix_into_f32 = network.find_imported_def("load_matrix_into_f32")
    load_matrix_into_i64 = network.find_imported_def("load_matrix_into_i64")

    #mat_reshape = network.find_imported_def("matrix_reshape_f32")
    #mat_reshape_const = network.find_imported_def("matrix_reshape_const_f32")

    mat_reshape_setup = network.find_imported_def("matrix_reshape_setup")
    mat_reshape_impl = network.find_imported_def("matrix_reshape_impl")

    mat_mul_setup = network.find_imported_def("matrix_multiply_setup")
    mat_mul_impl = network.find_imported_def("matrix_multiply_impl")
    mat_add_setup = network.find_imported_def("matrix_add_setup")
    mat_add_impl = network.find_imported_def("matrix_add_impl")
    mat_conv_exec = network.find_imported_def("matrix_convolution_padded_impl")
    mat_conv_setup = network.find_imported_def("matrix_convolution_padded_setup")
    mat_gemm_setup = network.find_imported_def("matrix_gemm_setup")
    mat_gemm_impl = network.find_imported_def("matrix_gemm_impl")

    #mat_flatten = network.find_imported_def("matrix_flatten_f32")
    #mat_relu = network.find_imported_def("matrix_relu")
    #mat_lrn = network.find_imported_def("matrix_lrn_f32")
    #mat_softmax = network.find_imported_def("matrix_softmax")
    #mat_log_softmax = network.find_imported_def("matrix_log_softmax")
    #mat_max_pool = network.find_imported_def("matrix_max_pool")
    #mat_transpose = network.find_imported_def("matrix_transpose_f32")
    #mat_avg_pool = network.find_imported_def("matrix_avg_pool")
    #mat_dropout = network.find_imported_def("matrix_dropout_f32")
    #mat_concat = network.find_imported_def("matrix_concat4_f32")

    tensor_return_type = alloc_initializer_f32_thorin.type.args[-1]
    tensor_type = tensor_return_type.args[1]

    tensor_i64_return_type = alloc_initializer_i64_thorin.type.args[-1]
    tensor_i64_type = tensor_i64_return_type.args[1]

    network_exec_type = ThorinFnType([mem_type, tensor_type], tensor_type)
    setup_exec_type = ThorinFnType([mem_type], network_exec_type);

    input_name = graph.input[0].name
    output_name = graph.output[0].name

    # Node format: { "result": <whatever this node produces>
    #                "call": lambda in_cont, in_mem <used when the node is called from somewhere>
    #                "cont": lambda out_cont, *out_param <used when the node is supposed to call something else>
    #                "block": (cont, mem) <the last cont and memory of this node, used similar to "cont".
    #}
    nodes = {}

    unordered_nodes = []
    ordered_nodes = []
    initializer_nodes = []

    def link_nodes(entry, exit):
        cont, mem = entry["block"]
        exit["call"](cont, mem)

    with ThorinContinuation(setup_exec_type, internal="setup_network", thorin=network) as (setup_network, setup_network_mem, setup_ret):
        for initializer in graph.initializer:
            initializer_nodes.append(load_initializer(initializer))

        with ThorinContinuation(network_exec_type) as (run_network, run_network_mem, run_image, ret_function):
            buildImage_continue, buildImage_mem = ThorinContinuation(return_type).__enter__()

            nodes[input_name] = {"result": run_image,
                                 "call": lambda in_cont, in_mem: in_cont(buildImage_continue, in_mem),
                                 "cont": lambda out_cont, *out_param: buildImage_continue(out_cont, buildImage_mem, *out_param),
                                 "block": (buildImage_continue, buildImage_mem)}

            ordered_nodes.append(input_name)

            for node in graph.node:
                exec_node, init_node = build_node(node)
                if exec_node is not None:
                    ordered_nodes.append(exec_node)
                if init_node is not None:
                    initializer_nodes.append(init_node);

            for i in range(0, len(ordered_nodes) - 1):
                link_nodes(nodes[ordered_nodes[i]], nodes[ordered_nodes[i+1]])

            #Execute entry
            nodes[ordered_nodes[0]]["call"](run_network, run_network_mem)

            #Execute exit
            nodes[output_name]["cont"](ret_function, nodes[output_name]["result"])

        for i in range(0, len(initializer_nodes) - 1):
            link_nodes(nodes[initializer_nodes[i]], nodes[initializer_nodes[i+1]])

        if initializer_nodes == []:
            setup_network(setup_ret, setup_network_mem, run_network);
            assert(False);
        else:
            nodes[initializer_nodes[0]]["call"](setup_network, setup_network_mem)

            nodes[initializer_nodes[-1]]["cont"](setup_ret, run_network)

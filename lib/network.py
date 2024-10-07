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

f32_ptr_type = ThorinPointerType(f32_type)
iarrptr_type = ThorinPointerType(ThorinIndefiniteArrayType(i8_type))
data_type = ThorinPointerType(ThorinIndefiniteArrayType(f32_type))
i64_arr_type = ThorinPointerType(ThorinIndefiniteArrayType(i64_type))

return_type = ThorinFnType([mem_type])
network_exec_type = ThorinFnType([mem_type, data_type], data_type)
exec_type = ThorinFnType([mem_type], ThorinFnType([mem_type]))

thorin_constant_producer_type = ThorinFnType([mem_type, ThorinTupleType([i32_type, i64_arr_type])])


def alloc_tensor(entry_mem, passmanager, finish_cont, dimensions):
    #print("Alloc tensor with", dimensions)

    thorin_dimensions = list(map(lambda x: ThorinConstant(i64_type, x), dimensions))
    sizes = ThorinDefiniteArray(i64_type, thorin_dimensions)

    with ThorinContinuation(tensor_type.formated_args[2][1], filter=True) as (size_lambda, size_mem, dimension, size_return):
        r = ThorinExtract(sizes, dimension)
        size_lambda(size_return, size_mem, r)

    return (alloc_tensor_thorin, entry_mem, passmanager, ThorinConstant(i32_type, len(thorin_dimensions)), size_lambda, finish_cont)


def alloc_initializer(entry_mem, passmanager, finish_cont, dimensions):
    #print("Alloc initializer with", dimensions)
    thorin_dimensions = list(map(lambda x: ThorinConstant(i64_type, x), dimensions))
    sizes = ThorinDefiniteArray(i64_type, thorin_dimensions)

    local_tensor_type = finish_cont.type.args[1]

    with ThorinContinuation(local_tensor_type.formated_args[2][1], filter=True) as (size_lambda, size_mem, dimension, size_return):
        r = ThorinExtract(sizes, dimension)
        size_lambda(size_return, size_mem, r)

    if local_tensor_type == tensor_i64_type:
        alloc_function = alloc_initializer_i64_thorin
    else:
        alloc_function = alloc_initializer_f32_thorin

    return (alloc_function, entry_mem, passmanager, ThorinConstant(i32_type, len(dimensions)), size_lambda, finish_cont)


def alloc_and_load_tensor(entry_mem, passmanager, finish_cont, dimensions, matrix_name):
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

    return alloc_initializer(entry_mem, passmanager, alloc_continue, dimensions)
    #return alloc_initializer(entry_mem, passmanager, finish_cont, dimensions) #Do not load stuf, only allocation.


def convert_to_global_array(input_array, thorin_type):
    thorin_array = ThorinDefiniteArray(thorin_type, [ThorinConstant(thorin_type, x) for x in input_array])
    glob = ThorinGlobal(thorin_array)
    return ThorinBitcast(glob, ThorinPointerType(ThorinIndefiniteArrayType(thorin_type)))


with Thorin("network") as network:
    network.include(NETWORK_TOOLS_PATH)

    sequential = network.find_imported_def("sequential")

    build_tensor_thorin = network.find_imported_def("build_tensor_f32")

    alloc_tensor_thorin = network.find_imported_def("alloc_tensor_f32")
    alloc_initializer_f32_thorin = network.find_imported_def("alloc_initializer_f32")
    alloc_initializer_i64_thorin = network.find_imported_def("alloc_initializer_i64")
    load_matrix_into_f32 = network.find_imported_def("load_matrix_into_f32")
    load_matrix_into_i64 = network.find_imported_def("load_matrix_into_i64")

    mat_flatten = network.find_imported_def("matrix_flatten_f32")
    #mat_mul = network.find_imported_def("matrix_multiply")
    mat_add = network.find_imported_def("matrix_add")
    mat_relu = network.find_imported_def("matrix_relu")
    #mat_lrn = network.find_imported_def("matrix_lrn_f32")
    mat_gemm = network.find_imported_def("matrix_gemm_f32")
    #mat_softmax = network.find_imported_def("matrix_softmax")
    mat_log_softmax = network.find_imported_def("matrix_log_softmax")
    mat_reshape = network.find_imported_def("matrix_reshape_f32")
    mat_reshape_const = network.find_imported_def("matrix_reshape_const_f32")
    mat_conv = network.find_imported_def("matrix_convolution_padded")
    mat_max_pool = network.find_imported_def("matrix_max_pool")
    mat_transpose = network.find_imported_def("matrix_transpose_f32")
    #mat_avg_pool = network.find_imported_def("matrix_avg_pool")
    #mat_dropout = network.find_imported_def("matrix_dropout_f32")

    #mat_concat = network.find_imported_def("matrix_concat4_f32")

    body_type = sequential.type.args[1]
    passmanager_type = body_type.args[1]

    tensor_return_type = alloc_initializer_f32_thorin.type.args[-1]
    tensor_type = tensor_return_type.args[1]

    tensor_i64_return_type = alloc_initializer_i64_thorin.type.args[-1]
    tensor_i64_type = tensor_i64_return_type.args[1]

    buffer_type = tensor_type.formated_args[0][1]

    with ThorinContinuation(network_exec_type, internal="run_network", thorin=network) as (run_network, run_network_mem, image, ret_function):
        run_network_mem, frame = thorinEnterExtract(run_network_mem)
        result_ptr = ThorinSlot(frame, data_type)

        with ThorinContinuation(body_type, filter=True) as (body_fn, body_mem, passmanager, body_return):
            # Node format: { "result": <whatever this node produces>
            #                "call": lambda in_cont, in_mem <used when the node is called from somewhere>
            #                "cont": lambda out_cont, *out_param <used when the node is supposed to call something else>
            #                "block": (cont, mem) <the last cont and memory of this node, used similar to "cont".
            #}
            nodes = {}

            def translate_operation(onnx_node):
                if onnx_node.op_type == "Gemm" or "Linear" in onnx_node.op_type:
                    return mat_gemm
                elif onnx_node.op_type == "Flatten" or "Flatten" in onnx_node.op_type:
                    return mat_flatten
                elif onnx_node.op_type == "Add" or "Add" in onnx_node.op_type:
                    return mat_add
                elif onnx_node.op_type == "Reshape" or "_view" in onnx_node.op_type:
                    input_size = nodes[onnx_node.input[1]]["onnx_node"]
                    if hasattr(input_size, "op_type") and input_size.op_type == "Constant":
                        return mat_reshape_const
                    else:
                        return mat_reshape
                elif onnx_node.op_type == "Relu" or "relu" in onnx_node.op_type:
                    return mat_relu
                elif onnx_node.op_type == "LRN" or "LocalResponseNorm" in onnx_node.op_type:
                    return mat_lrn
                elif onnx_node.op_type == "LogSoftmax" or "log_softmax" in onnx_node.op_type:
                    return mat_log_softmax
                elif onnx_node.op_type == "Softmax" or "softmax" in onnx_node.op_type:
                    return mat_softmax
                elif onnx_node.op_type == "Conv" or "conv" in onnx_node.op_type:
                    return mat_conv
                elif onnx_node.op_type == "MaxPool" or "max_pool" in onnx_node.op_type:
                    return mat_max_pool
                elif onnx_node.op_type == "Transpose":
                    return mat_transpose
                elif onnx_node.op_type == "AveragePool" or "avg_pool" in onnx_node.op_type:
                    return mat_avg_pool
                elif onnx_node.op_type == "Dropout" or "dropout" in onnx_node.op_type:
                    return mat_dropout
                elif onnx_node.op_type == "Concat" or "SequenceConstruct" in onnx_node.op_type:
                    return mat_concat
                elif "aten_cat" in onnx_node.op_type:
                    return mat_dropout
                else:
                    print("op unknown:", onnx_node.op_type, "at", onnx_node.name)
                    print(onnx_node)
                    assert(False)

            def load_initializer(onnx_node):
                dimensions = onnx_node.dims
                data_type = onnx_node.data_type

                if data_type == 7:
                    local_tensor_return_type = tensor_i64_return_type
                else:
                    local_tensor_return_type = tensor_return_type

                with ThorinContinuation(local_tensor_return_type) as (return_cont, return_mem, result_tensor):
                    #dimensions.reverse()
                    call_function = lambda in_cont, in_mem: in_cont(*alloc_and_load_tensor(in_mem, passmanager, return_cont, dimensions, onnx_node.name))
                    return_function = lambda out_cont, *out_param: return_cont(out_cont, return_mem, *out_param)

                    update_node = {"result": result_tensor, "call": call_function, "cont": return_function, "block": (return_cont, return_mem), "onnx_node": onnx_node}
                    nodes.update({onnx_node.name : update_node})
                    return onnx_node.name

            def build_node(onnx_node):
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

                #Used to gather additional attributes that might be needed, e.g. padding
                def get_attributes(*args):
                    results = [None for _ in range(0, len(args))]
                    for attribute in onnx_node.attribute:
                        for index, arg in zip(range(0, len(args)), args):
                            if attribute.name == arg:
                                results[index] = attribute.ints
                    return results

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
                        return onnx_node.output[0]

                elif onnx_node.op_type == "Transpose":
                    perm, = get_attributes("perm")

                    perm_global = convert_to_global_array(perm, i64_type)
                    output_shape_global = convert_to_global_array(output_shape, i64_type)

                    attributes = [perm_global, output_shape_global]

                    thorin_operation = translate_operation(onnx_node)
                    return_fn_type = thorin_operation.type.args[-1]

                    with ThorinContinuation(return_fn_type) as (return_cont, return_mem, result_tensor):
                        #call_function = lambda in_cont, in_mem: in_cont(specialized_operation, in_mem, passmanager, *[nodes[name]["result"] for name in onnx_node.input], return_cont)
                        call_function = lambda in_cont, in_mem: in_cont(thorin_operation, in_mem, passmanager, *[nodes[name]["result"] for name in onnx_node.input], *attributes, return_cont)
                        return_function = lambda out_cont, *out_param: return_cont(out_cont, return_mem, *out_param)

                        update_node = {"result": result_tensor, "call": call_function, "cont": return_function, "block": (return_cont, return_mem), "onnx_node": onnx_node}
                        nodes.update({onnx_node.output[0] : update_node})
                        return onnx_node.output[0]

                elif onnx_node.op_type == "Conv" or "conv" in onnx_node.op_type or onnx_node.op_type == "MaxPool" or "max_pool" in onnx_node.op_type or onnx_node.op_type == "AveragePool":
                    shape, strides, padding = get_attributes("kernel_shape", "strides", "pads")

                    shape_global = convert_to_global_array(shape, i64_type)
                    stride_global = convert_to_global_array(strides, i64_type)
                    padding_global = convert_to_global_array(padding, i64_type)
                    output_shape_global = convert_to_global_array(output_shape, i64_type)

                    attributes = [output_shape_global, shape_global, stride_global, padding_global]

                    thorin_operation = translate_operation(onnx_node)
                    return_fn_type = thorin_operation.type.args[-1]

                    #with ThorinContinuation(ThorinFnType([mem_type, passmanager_type, tensor_type, tensor_type, tensor_1_type, return_fn_type])) as (specialized_operation, specialized_mem, pm, input_tensor, weight_tensor, bias_tensor, return_cont):
                    #    specialized_operation(thorin_operation, specialized_mem, pm, input_tensor, weight_tensor, bias_tensor, *attributes, return_cont)

                    with ThorinContinuation(return_fn_type) as (return_cont, return_mem, result_tensor):
                        #call_function = lambda in_cont, in_mem: in_cont(specialized_operation, in_mem, passmanager, *[nodes[name]["result"] for name in onnx_node.input], return_cont)
                        call_function = lambda in_cont, in_mem: in_cont(thorin_operation, in_mem, passmanager, *[nodes[name]["result"] for name in onnx_node.input], *attributes, return_cont)
                        return_function = lambda out_cont, *out_param: return_cont(out_cont, return_mem, *out_param)

                        update_node = {"result": result_tensor, "call": call_function, "cont": return_function, "block": (return_cont, return_mem), "onnx_node": onnx_node}
                        nodes.update({onnx_node.output[0] : update_node})
                        return onnx_node.output[0]

                else:
                    thorin_operation = translate_operation(onnx_node)
                    return_fn_type = thorin_operation.type.args[-1]

                    output_shape_global = convert_to_global_array(output_shape, i64_type)

                    attributes = [output_shape_global]

                    with ThorinContinuation(return_fn_type) as (return_cont, return_mem, result_tensor):
                        call_function = lambda in_cont, in_mem: in_cont(thorin_operation, in_mem, passmanager, *[nodes[name]["result"] for name in onnx_node.input], *attributes, return_cont)
                        return_function = lambda out_cont, *out_param: return_cont(out_cont, return_mem, *out_param)

                        update_node = {"result": result_tensor, "call": call_function, "cont": return_function, "block": (return_cont, return_mem), "onnx_node": onnx_node}
                        nodes.update({onnx_node.output[0] : update_node})
                        return onnx_node.output[0]

            unordered_nodes = []
            ordered_nodes = []
            initializer_nodes = []

            input_name = graph.input[0].name
            output_name = graph.output[0].name

            image_dims = list(x.dim_value for x in graph.input[0].type.tensor_type.shape.dim)

            num_image_dims = len(image_dims)
            thorin_dimensions = list(map(lambda x: ThorinConstant(i64_type, x), image_dims))
            sizes = ThorinDefiniteArray(i64_type, thorin_dimensions)

            #size_lambda = ...
            with ThorinContinuation(tensor_type.formated_args[2][1], filter=True) as (size_lambda, size_mem, dimension, size_return):
                r = ThorinExtract(sizes, dimension)
                size_lambda(size_return, size_mem, r)

            image_buffer = ThorinStruct(buffer_type, [ThorinBitcast(image, iarrptr_type), ThorinConstant(i64_type, 0), ThorinConstant(i32_type, 0)])

            buildImage_continue, buildImage_mem, tensorImage = ThorinContinuation(tensor_return_type).__enter__()

            nodes[input_name] = {"result": tensorImage,
                                 "call": lambda in_cont, in_mem: in_cont(build_tensor_thorin, in_mem, image_buffer, ThorinConstant(i32_type, num_image_dims), size_lambda, buildImage_continue),
                                 "cont": lambda out_cont, *out_param: buildImage_continue(out_cont, buildImage_mem, *out_param),
                                 "block": (buildImage_continue, buildImage_mem)}

            ordered_nodes.append(input_name)

            #num_initializers = 0
            num_initializers = len(graph.initializer) + 1 #?

            always_build_initializers = True
            if always_build_initializers:
                for initializer in graph.initializer:
                    initializer_name = load_initializer(initializer)
                    ordered_nodes.append(initializer.name)
                for node in graph.node:
                    node_output = build_node(node)
                    required_nodes = node.input
                    unordered_nodes.append((node.output[0], required_nodes))

                for node, required in unordered_nodes:
                    my_required = list(required)

                    for node_init in initializer_nodes:
                        if node_init in my_required:
                            ordered_nodes.insert(0, node_init)
                            num_initializers += 1

                    for node_known in ordered_nodes:
                        while node_known in initializer_nodes:
                            initializer_nodes.remove(node_known)

                    for i in range(0, len(ordered_nodes)):
                        while ordered_nodes[i] in my_required:
                            my_required.remove(ordered_nodes[i])
                        if my_required == []:
                            insert_index = i + 1
                            if insert_index < num_initializers:
                                insert_index = num_initializers
                            ordered_nodes.insert(insert_index, node)
                            break
                    else:
                        print(ordered_nodes)
                        print(my_required)
                        assert(False)

                    if node == output_name:
                        print("Added output node to ordered_nodes, finished")
                        break;
            else:
                todo = [output_name]
                initializer_index = 0

                graph_node_names = []
                for node in graph.node:
                    graph_node_names.append(node.output[0])
                initializer_node_names = []
                for node in graph.initializer:
                    initializer_node_names.append(node.name)

                print("Graph:", graph_node_names)
                print("Init:", initializer_node_names)

                while not todo == []:
                    node = todo.pop()

                    print(node)

                    if node in graph_node_names:
                        index = graph_node_names.index(node)
                        onnx_node = graph.node[index]

                        node_output = build_node(onnx_node)
                        ordered_nodes.insert(initializer_index, node)

                        required_nodes = list(onnx_node.input)

                        for req_node in required_nodes:
                            if(req_node in ordered_nodes):
                                print("Error")
                                print(req_node)
                            assert(req_node not in ordered_nodes)

                            if req_node in todo:
                                todo.remove(req_node)

                        for req_node in required_nodes:
                            todo.insert(0, req_node)

                    elif node in initializer_node_names:
                        index = initializer_node_names.index(node)
                        onnx_node = graph.initializer[index]

                        initializer_name = load_initializer(onnx_node)
                        ordered_nodes.insert(initializer_index, node)
                        initializer_index += 1

                    else:
                        assert(False)

            #TODO: emit nodes based on requirements, no static ordering. This way, we can easily prune nodes.
            #print("ordereing")
            #for n in ordered_nodes:
            #    print(n)

            def link_nodes(entry, exit):
                cont, mem = entry["block"]
                exit["call"](cont, mem)

            for i in range(0, len(ordered_nodes) - 1):
                link_nodes(nodes[ordered_nodes[i]], nodes[ordered_nodes[i+1]])
                if ordered_nodes[i+1] == output_name:
                    print("Emitted link to output node, everything else is not needed.")
                    break

            #Execute entry
            nodes[ordered_nodes[0]]["call"](body_fn, body_mem)

            #Execute exit
            with ThorinContinuation(tensor_return_type) as (result_store_cont, return_mem, result_tensor):
                result_buffer = ThorinExtract(result_tensor, 0)
                result_data = ThorinBitcast(ThorinExtract(result_buffer, 0), data_type)
                return_mem = return_mem << (result_ptr, result_data)
                result_store_cont(body_return, return_mem)

            nodes[output_name]["cont"](result_store_cont, nodes[output_name]["result"])

        #Support code do generate a manager.
        with ThorinContinuation(return_type) as (return_cont, return_mem):
            return_mem, result_data = return_mem >> result_ptr
            return_cont(ret_function, return_mem, result_data)

        with ThorinContinuation(exec_type) as (exec_cont, exec_mem, exec_int):
            exec_cont(exec_int, exec_mem, return_cont)

        run_network(sequential, run_network_mem, body_fn, exec_cont)
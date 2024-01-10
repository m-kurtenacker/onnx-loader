import onnx

#ONNX_MODEL = "mnist-example/mnist.onnx"

#ONNX_MODEL = "../mnist-example/mnist_linear.onnx"
#MAGIC_CONST = 784

ONNX_MODEL = "../mnist-example/mnist_cnn.onnx"
MAGIC_CONST = 576

model = onnx.load(ONNX_MODEL)
graph = model.graph

from pythorin import *


mem_type = ThorinMemType()
f32_type = ThorinPrimType("qf32")
i32_type = ThorinPrimType("qs32")
i64_type = ThorinPrimType("qs64")
i8_type = ThorinPrimType("qs8")
u8_type = ThorinPrimType("qu8")

f32_ptr_type = ThorinPointerType(f32_type)
string_type = ThorinPointerType(ThorinIndefiniteArrayType(ThorinPrimType("pu8")))
iarrptr_type = ThorinPointerType(ThorinIndefiniteArrayType(i8_type))
data_type = ThorinPointerType(ThorinIndefiniteArrayType(f32_type))

buffer_type = ThorinStructType("Buffer", [("data", iarrptr_type), ("size", i64_type), ("device", i32_type)])

alloc_type = ThorinFnType([mem_type, i64_type], buffer_type)
release_type = ThorinFnType([mem_type, buffer_type], True)
passmanager_type = ThorinStructType("PassManager", [("alloc", alloc_type), ("release", release_type)])

size_fn_type = ThorinFnType([mem_type, i32_type], i64_type)
access_fn_type = ThorinFnType([mem_type, ThorinPointerType(ThorinIndefiniteArrayType(i64_type))], f32_ptr_type)
tensor_type = ThorinStructType("Tensor_f32", [("buffer", buffer_type), ("num_dims", i32_type), ("size_fn", size_fn_type), ("access_fn", access_fn_type)])

body_type = ThorinFnType([mem_type, passmanager_type], True)

ret_type = ThorinFnType([mem_type, tensor_type])
return_type = ThorinFnType([mem_type])
network_exec_type = ThorinFnType([mem_type, data_type], data_type)
exec_type = ThorinFnType([mem_type], ThorinFnType([mem_type]))
tensor_return_type = ThorinFnType([mem_type, tensor_type])
access_return_type = ThorinFnType([mem_type, f32_ptr_type])


def alloc_tensor(entry_mem, passmanager, finish_cont, dimensions):
    print("Alloc tensor with", dimensions)
    thorin_dimensions = list(map(lambda x: ThorinConstant(i64_type, x), dimensions))
    with ThorinContinuation(size_fn_type, filter=True) as (size_lambda, size_mem, dimension, size_return):
        sizes = ThorinDefiniteArray(i64_type, thorin_dimensions)
        size = ThorinExtract(sizes, dimension)
        size_lambda(size_return, size_mem, size)

    num_dimensions = len(dimensions)
    return (alloc_tensor_thorin, entry_mem, passmanager, ThorinConstant(i32_type, num_dimensions), size_lambda, finish_cont)


def alloc_and_load_tensor(entry_mem, passmanager, finish_cont, dimensions, matrix_name):
    model_name = thorinString(ONNX_MODEL)
    matrix_name = thorinString(matrix_name)

    with ThorinContinuation(tensor_return_type) as (alloc_continue, alloc_mem, tensor):
        with ThorinContinuation(return_type) as (load_cont, finish_mem):
            load_cont(finish_cont, finish_mem, tensor)
        with ThorinContinuation(exec_type, filter=True) as (load_return, load_mem, load_int):
            load_return(load_int, load_mem, load_cont);

        alloc_continue(load_matrix, alloc_mem, tensor, model_name, matrix_name, load_return)

    return alloc_tensor(entry_mem, passmanager, alloc_continue, dimensions)


with Thorin("network") as network:
    network.include("network-tools.thorin.json")

    sequential = network.find_imported_def("sequential")
    alloc_tensor_thorin = network.find_imported_def("alloc_tensor_f32")
    load_matrix = network.find_imported_def("load_matrix_into")
    mat_flatten = network.find_imported_def("matrix_flatten_f32")
    mat_mul = network.find_imported_def("matrix_multiply_f32")
    mat_add = network.find_imported_def("matrix_add_f32")
    mat_relu = network.find_imported_def("matrix_relu_f32")
    mat_gemm = network.find_imported_def("matrix_gemm_f32")
    mat_softmax = network.find_imported_def("matrix_softmax_f32")
    mat_log_softmax = network.find_imported_def("matrix_log_softmax_f32")
    mat_reshape = network.find_imported_def("matrix_reshape_f32")
    mat_conv = network.find_imported_def("matrix_convolution_f32")
    mat_max_pool = network.find_imported_def("matrix_max_pool_f32")
    mat_dropout = network.find_imported_def("matrix_dropout_f32")

    #Define types for imported continuations. Only needed for printing.
    sequential.type = ThorinFnType([mem_type, body_type], ThorinFnType([mem_type], True))
    alloc_tensor_thorin.type = ThorinFnType([mem_type, passmanager_type, i32_type, size_fn_type], tensor_type)
    load_matrix.type = ThorinFnType([mem_type, tensor_type, string_type, string_type], ThorinFnType([mem_type], True))
    mat_flatten.type = ThorinFnType([mem_type, passmanager_type, tensor_type], tensor_type)
    mat_mul.type = ThorinFnType([mem_type, passmanager_type, tensor_type, tensor_type], tensor_type)
    mat_add.type = ThorinFnType([mem_type, passmanager_type, tensor_type, tensor_type], tensor_type)
    mat_relu.type = ThorinFnType([mem_type, passmanager_type, tensor_type], tensor_type)
    mat_gemm.type = ThorinFnType([mem_type, passmanager_type, tensor_type, tensor_type, tensor_type], tensor_type)
    mat_softmax.type = ThorinFnType([mem_type, passmanager_type, tensor_type], tensor_type)
    mat_log_softmax.type = ThorinFnType([mem_type, passmanager_type, tensor_type], tensor_type)
    mat_dropout.type = ThorinFnType([mem_type, passmanager_type, tensor_type], tensor_type)
    mat_max_pool.type = ThorinFnType([mem_type, passmanager_type, tensor_type], tensor_type)
    mat_conv.type = ThorinFnType([mem_type, passmanager_type, tensor_type, tensor_type, tensor_type], tensor_type)

    with ThorinContinuation(network_exec_type, internal="run_network", thorin=network) as (run_network, run_network_mem, image, ret_function):
        run_network_mem, frame = thorinEnterExtract(run_network_mem)
        result_ptr = ThorinSlot(frame, data_type)

        with ThorinContinuation(body_type, filter=True) as (body_fn, body_mem, passmanager, body_return):
            nodes = {}

            def translate_operation(onnx_node):
                if onnx_node.op_type == "Gemm" or "Linear" in onnx_node.op_type:
                    return mat_gemm
                elif onnx_node.op_type == "Flatten" or "Flatten" in onnx_node.op_type:
                    return mat_flatten
                elif onnx_node.op_type == "Reshape" or "_view" in onnx_node.op_type:
                    return mat_reshape
                elif onnx_node.op_type == "Relu" or "relu" in onnx_node.op_type:
                    return mat_relu
                elif onnx_node.op_type == "Log_Softmax" or "log_softmax" in onnx_node.op_type:
                    return mat_log_softmax
                elif onnx_node.op_type == "Softmax" or "softmax" in onnx_node.op_type:
                    return mat_softmax
                elif onnx_node.op_type == "Conv" or "conv" in onnx_node.op_type:
                    return mat_conv
                elif onnx_node.op_type == "MaxPool" or "max_pool" in onnx_node.op_type:
                    return mat_max_pool
                elif onnx_node.op_type == "Dropout" or "dropout" in onnx_node.op_type:
                    return mat_dropout
                else:
                    print("op unknown:", onnx_node.op_type, "at", onnx_node.name)
                    print(onnx_node)
                    assert(False)

            def load_initializer(onnx_node):
                with ThorinContinuation(tensor_return_type) as (return_cont, return_mem, result_tensor):
                    dimensions = onnx_node.dims
                    dimensions.reverse()

                    call_function = lambda in_cont, in_mem: in_cont(*alloc_and_load_tensor(in_mem, passmanager, return_cont, dimensions, onnx_node.name))
                    return_function = lambda out_cont, *out_param: return_cont(out_cont, return_mem, *out_param)

                    update_node = {"result": result_tensor, "call": call_function, "cont": return_function, "block": (return_cont, return_mem, result_tensor)}
                    nodes.update({onnx_node.name : update_node})
                    return onnx_node.name

            def build_node(onnx_node):
                if onnx_node.op_type == "Constant":
                    assert(False)
                    with ThorinContinuation(ThorinFnType([mem_type, ThorinTupleType([i32_type, ThorinPointerType(ThorinIndefiniteArrayType(i64_type))])])) as (return_cont, return_mem, result_tuple):
                        addr_ptr = ThorinSlot(frame, ThorinDefiniteArrayType(i64_type, 1))
                        addr_ptr_opaque = ThorinBitcast(addr_ptr, ThorinPointerType(ThorinIndefiniteArrayType(i64_type)))

                        return_mem = return_mem << (addr_ptr, ThorinDefiniteArray(i64_type, [ThorinConstant(i64_type, MAGIC_CONST)]))

                        #print("Constant:")
                        #print(onnx_node)

                        shape_tuple = ThorinTuple([ThorinConstant(i32_type, 1), addr_ptr_opaque]);

                        call_function = lambda in_cont, in_mem: in_cont(return_cont, in_mem, shape_tuple)
                        return_function = lambda out_cont, *out_param: return_cont(out_cont, return_mem, *out_param)

                        update_node = {"result": result_tuple, "call": call_function, "cont": return_function, "block": (return_cont, return_mem, result_tuple)}
                        nodes.update({onnx_node.output[0] : update_node})
                        return onnx_node.output[0]
                elif onnx_node.op_type == "Reshape" or "_view" in onnx_node.op_type:
                    with ThorinContinuation(tensor_return_type) as (return_cont, return_mem, result_tensor):
                        call_function = lambda in_cont, in_mem: in_cont(mat_flatten, in_mem, passmanager, *[nodes[name]["result"] for name in onnx_node.input[:-1]], return_cont)
                        return_function = lambda out_cont, *out_param: return_cont(out_cont, return_mem, *out_param)

                        update_node = {"result": result_tensor, "call": call_function, "cont": return_function, "block": (return_cont, return_mem, result_tensor)}
                        nodes.update({onnx_node.output[0] : update_node})
                        return onnx_node.output[0]
                else:
                    with ThorinContinuation(tensor_return_type) as (return_cont, return_mem, result_tensor):
                        call_function = lambda in_cont, in_mem: in_cont(translate_operation(onnx_node), in_mem, passmanager, *[nodes[name]["result"] for name in onnx_node.input], return_cont)
                        return_function = lambda out_cont, *out_param: return_cont(out_cont, return_mem, *out_param)

                        update_node = {"result": result_tensor, "call": call_function, "cont": return_function, "block": (return_cont, return_mem, result_tensor)}
                        nodes.update({onnx_node.output[0] : update_node})
                        return onnx_node.output[0]

            def link_nodes(entry, exit):
                cont, mem, _ = entry["block"]
                exit["call"](cont, mem)

            unordered_nodes = []
            ordered_nodes = []

            input_name = graph.input[0].name
            output_name = graph.output[0].name
            #output_name = "relu_1"
            #output_name = "conv1_1"

            local_copy = False
            if local_copy:
                assert(False) # This codepath is outdated and needs to be fixed to support image chanels.
                alloc_block = ()
                allocImage_continue, allocImage_mem, tensorImage = ThorinContinuation(tensor_return_type).__enter__()

                def rangeX_body(entry_block, mem, indexX, continueX_block):
                    def rangeY_body(entry_block, mem, indexY, continueY_block):
                        image_x_y_ptr = ThorinLEA([image, indexY * 28 + indexX])

                        #mem, frame = thorinEnterExtract(mem)
                        addr_ptr = ThorinSlot(frame, ThorinDefiniteArrayType(i64_type, 2))
                        addr_ptr_opaque = ThorinBitcast(addr_ptr, ThorinPointerType(ThorinIndefiniteArrayType(i64_type)))

                        mem = mem << (addr_ptr, ThorinDefiniteArray(i64_type, [ThorinCast(indexX, i64_type), ThorinCast(indexY, i64_type)]))

                        image_access_fn = ThorinExtract(tensorImage, 3)

                        with ThorinContinuation(access_return_type) as (store_cont, store_mem, store_ptr):
                            store_mem, value = store_mem >> image_x_y_ptr
                            store_mem = store_mem << (store_ptr, value)

                            store_cont(continueY_block, store_mem)

                        entry_block(image_access_fn, mem, addr_ptr_opaque, store_cont)

                    def rangeY_return(return_block, mem):
                        return_block(continueX_block, mem)

                    entry_block(*thorinRangeFn(mem, 0, 28, 1, rangeY_body, rangeY_return))

                def rangeX_return(return_block, mem):
                    global alloc_block
                    alloc_block = (return_block, mem, tensorImage)

                allocImage_continue(*thorinRangeFn(allocImage_mem, 0, 28, 1, rangeX_body, rangeX_return))

                nodes[input_name] = {"result": tensorImage,
                                     "call": lambda in_cont, in_mem: in_cont(*alloc_tensor(in_mem, passmanager, allocImage_continue, [28, 28])),
                                     "cont": lambda out_cont, *out_param: alloc_block[0](out_cont, alloc_block[1], *out_param),
                                     "block": alloc_block}

                ordered_nodes.append(input_name)
            else:
                image_dims = [28, 28, 1]
                num_image_dims = len(image_dims)
                with ThorinContinuation(size_fn_type, filter=True) as (size_lambda, size_mem, index, size_return):
                    thorin_dimensions = list(map(lambda x: ThorinConstant(i64_type, x), image_dims))
                    sizes = ThorinDefiniteArray(i64_type, thorin_dimensions)

                    with ThorinContinuation(size_fn_type, filter=True) as (size_lambda, size_mem, dimension, size_return):
                        size = ThorinExtract(sizes, dimension)
                        size_lambda(size_return, size_mem, size)

                with ThorinContinuation(access_fn_type, filter=True) as (access_lambda, access_mem, dimensions_ptr, access_return):
                    x_ptr = ThorinLEA([dimensions_ptr, ThorinConstant(i64_type, 0)])
                    y_ptr = ThorinLEA([dimensions_ptr, ThorinConstant(i64_type, 1)])
                    #chan_ptr = ThorinLEA([dimensions_ptr, ThorinConstant(i64_type, 2)])
                    access_mem, x = access_mem >> x_ptr
                    access_mem, y = access_mem >> y_ptr
                    #access_mem, chan = access_mem >> chan_ptr

                    #image_chan_x_y_ptr = ThorinLEA([image, chan * ThorinConstant(i64_type, 28*28) + y * ThorinConstant(i64_type, 28) + x])
                    image_x_y_ptr = ThorinLEA([image, y * ThorinConstant(i64_type, 28) + x])

                    access_lambda(access_return, access_mem, image_x_y_ptr)
                image_buffer = ThorinStruct(buffer_type, [ThorinBitcast(image, iarrptr_type), ThorinConstant(i64_type, 0), ThorinConstant(i32_type, 0)])
                tensorImage = ThorinStruct(tensor_type, [image_buffer, ThorinConstant(i32_type, num_image_dims), size_lambda, access_lambda])
                nodes[input_name] = {"result": tensorImage}

            num_initializers = len(graph.initializer)

            for initializer in graph.initializer:
                initializer_name = load_initializer(initializer)
                print("Init:", initializer_name)
                ordered_nodes.append(initializer.name)
            for node in graph.node:
                if node.op_type == "Constant":
                    continue
                node_output = build_node(node)
                print("Node:", node_output)
                required_nodes = node.input
                unordered_nodes.append((node.output[0], required_nodes))
                if node_output == output_name:
                    print("Build end")
                    break

            print("ordereing")
            #TODO: emit nodes based on requirements, no static ordering. This way, we can easily prune nodes.

            for node, required in unordered_nodes:
                my_required = list(required)

                if not local_copy:
                    if input_name in my_required:
                        my_required.remove(input_name)
                
                if "_val_12" in required:
                    my_required.remove("_val_12")

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
                    print(my_required)
                    assert(False)

            for n in ordered_nodes:
                print(n)

            for i in range(0, len(ordered_nodes) - 1):
                link_nodes(nodes[ordered_nodes[i]], nodes[ordered_nodes[i+1]])
                #if ordered_nodes[i] == output_name:
                #    print("Link end")
                #    break

            #Execute entry
            nodes[ordered_nodes[0]]["call"](body_fn, body_mem)

            #Execute exit
            with ThorinContinuation(ret_type) as (result_store_cont, return_mem, result_tensor):
                result_buffer = ThorinExtract(result_tensor, 0)
                result_data = ThorinBitcast(ThorinExtract(result_buffer, 0), data_type)
                return_mem = return_mem << (result_ptr, result_data)
                result_store_cont(body_return, return_mem)

            nodes[output_name]["cont"](result_store_cont, nodes[output_name]["result"])

        with ThorinContinuation(return_type) as (return_cont, return_mem):
            return_mem, result_data = return_mem >> result_ptr
            return_cont(ret_function, return_mem, result_data)

        with ThorinContinuation(exec_type) as (exec_cont, exec_mem, exec_int):
            exec_cont(exec_int, exec_mem, return_cont)

        run_network(sequential, run_network_mem, body_fn, exec_cont)

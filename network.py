import onnx

ONNX_MODEL = "mnist-example/mnist.onnx"

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
image_type = ThorinPointerType(ThorinIndefiniteArrayType(ThorinPointerType(ThorinIndefiniteArrayType(f32_type))))

buffer_type = ThorinStructType("Buffer", [("data", iarrptr_type), ("size", i64_type), ("device", i32_type)])

alloc_type = ThorinFnType([mem_type, i64_type], buffer_type)
release_type = ThorinFnType([mem_type, buffer_type], True)
passmanager_type = ThorinStructType("PassManager", [("alloc", alloc_type), ("release", release_type)])

size_fn_type = ThorinFnType([mem_type, i32_type], i64_type)
access_fn_type = ThorinFnType([mem_type, ThorinPointerType(ThorinIndefiniteArrayType(i64_type))], f32_ptr_type)
tensor_type = ThorinStructType("Tensor_f32", [("buffer", buffer_type), ("num_dims", i32_type), ("size_fn", size_fn_type), ("access_fn", access_fn_type)])

body_type = ThorinFnType([mem_type, passmanager_type], True)

ret_type = ThorinFnType([mem_type, f32_type])
return_type = ThorinFnType([mem_type])
network_exec_type = ThorinFnType([mem_type, image_type, i32_type], f32_type)
exec_type = ThorinFnType([mem_type], ThorinFnType([mem_type]))
tensor_return_type = ThorinFnType([mem_type, tensor_type])
access_return_type = ThorinFnType([mem_type, f32_ptr_type])


def alloc_tensor(entry_mem, passmanager, finish_cont, dimensions):
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
    mat_softmax = network.find_imported_def("matrix_softmax_f32")
    mat_sparsecrossentropy = network.find_imported_def("matrix_sparsecrossentropy_f32")
    mat_flatten = network.find_imported_def("matrix_flatten_f32")
    mat_mul = network.find_imported_def("matrix_multiply_f32")
    mat_add = network.find_imported_def("matrix_add_f32")
    mat_relu = network.find_imported_def("matrix_relu_f32")
    mat_gemm = network.find_imported_def("matrix_gemm_f32")

    #Define types for imported continuations. Only needed for printing.
    #sequential.type = ThorinFnType([mem_type, body_type], ThorinFnType([mem_type], True))
    #alloc_tensor_thorin.type = ThorinFnType([mem_type, passmanager_type, i32_type, size_fn_type], tensor_type)
    #load_matrix.type = ThorinFnType([mem_type, tensor_type, string_type, string_type], ThorinFnType([mem_type], True))
    #mat_softmax.type = ThorinFnType([mem_type, passmanager_type, tensor_type], tensor_type)
    #mat_sparsecrossentropy.type = ThorinFnType([mem_type, passmanager_type, tensor_type, i32_type], f32_type)
    #mat_flatten.type = ThorinFnType([mem_type, passmanager_type, tensor_type], tensor_type)
    #mat_mul.type = ThorinFnType([mem_type, passmanager_type, tensor_type, tensor_type], tensor_type)
    #mat_add.type = ThorinFnType([mem_type, passmanager_type, tensor_type, tensor_type], tensor_type)
    #mat_relu.type = ThorinFnType([mem_type, passmanager_type, tensor_type], tensor_type)
    #mat_gemm.type = ThorinFnType([mem_type, passmanager_type, tensor_type, tensor_type, tensor_type], tensor_type)

    with ThorinContinuation(network_exec_type, internal="run_network", thorin=network) as (run_network, run_network_mem, image, label, ret_function):
        run_network_mem, frame = thorinEnterExtract(run_network_mem)
        error_accu = ThorinSlot(frame, f32_type)

        with ThorinContinuation(body_type, filter=True) as (body_fn, body_mem, passmanager, body_return):
            nodes = {}

            def translate_operation(onnx_node):
                if onnx_node.op_type == "Gemm":
                    return mat_gemm
                elif onnx_node.op_type == "Flatten":
                    return mat_flatten
                elif onnx_node.op_type == "Relu":
                    return mat_relu
                elif onnx_node.op_type == "Softmax":
                    return mat_softmax
                else:
                    assert(false)

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


            alloc_block = ()
            allocImage_continue, allocImage_mem, tensorImage = ThorinContinuation(tensor_return_type).__enter__()

            def rangeX_body(entry_block, mem, indexX, continueX_block):
                def rangeY_body(entry_block, mem, indexY, continueY_block):
                    mem, image_x = mem >> ThorinLEA([image, indexX])
                    image_x_y_ptr = ThorinLEA([image_x, indexY])

                    mem, frame = thorinEnterExtract(mem)
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

            nodes["image_input"] = {"result": tensorImage,
                                    "call": lambda in_cont, in_mem: in_cont(*alloc_tensor(in_mem, passmanager, allocImage_continue, [28, 28])),
                                    "cont": lambda out_cont, *out_param: alloc_block[0](out_cont, alloc_block[1], *out_param),
                                    "block": alloc_block}

            ordered_nodes.append("image_input")

            for initializer in graph.initializer:
                print(load_initializer(initializer))
                ordered_nodes.append(initializer.name)
            for node in graph.node:
                print(build_node(node))
                required_nodes = node.input
                unordered_nodes.append((node.output[0], required_nodes))
                #ordered_nodes.append(node.output[0])

            print("ordereing")

            for node, required in unordered_nodes:
                my_required = list(required)

                for i in range(0, len(ordered_nodes)):
                    while ordered_nodes[i] in my_required:
                        my_required.remove(ordered_nodes[i])
                    if my_required == []:
                        ordered_nodes.insert(i + 1, node)
                        break
                else:
                    print(my_required)
                    assert(False)

            for n in ordered_nodes:
                print(n)

            for i in range(0, len(ordered_nodes) - 1):
                link_nodes(nodes[ordered_nodes[i]], nodes[ordered_nodes[i+1]])

            #Execute entry
            nodes[ordered_nodes[0]]["call"](body_fn, body_mem)

            #Execute exit
            with ThorinContinuation(ret_type) as (error_store_cont, return_mem, value):
                return_mem = return_mem << (error_accu, value)
                error_store_cont(body_return, return_mem)

            softmaxReturnCont, softmax_mem, tensorSoftmax = ThorinContinuation(tensor_return_type).__enter__()

            nodes[graph.output[0].name]["cont"](mat_softmax, passmanager, nodes[graph.output[0].name]["result"], softmaxReturnCont)
            softmaxReturnCont(mat_sparsecrossentropy, softmax_mem, passmanager, tensorSoftmax, label, error_store_cont)

        with ThorinContinuation(return_type) as (return_cont, return_mem):
            return_mem, ret_value = return_mem >> error_accu
            return_cont(ret_function, return_mem, ret_value)

        with ThorinContinuation(exec_type) as (exec_cont, exec_mem, exec_int):
            exec_cont(exec_int, exec_mem, return_cont)

        run_network(sequential, run_network_mem, body_fn, exec_cont)

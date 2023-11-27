#import onnx

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
    model_name = thorinString("mnist-example/mnist.onnx")
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
    #network.include("mat.thorin.json")

    sequential = network.find_imported_def("sequential")
    sequential.type = ThorinFnType([mem_type, body_type], ThorinFnType([mem_type], True))

    alloc_tensor_thorin = network.find_imported_def("alloc_tensor_f32")
    alloc_tensor_thorin.type = ThorinFnType([mem_type, passmanager_type, i32_type, size_fn_type], tensor_type)

    load_matrix = network.find_imported_def("load_matrix_into")
    load_matrix.type = ThorinFnType([mem_type, tensor_type, string_type, string_type], ThorinFnType([mem_type], True))
    
    mat_softmax = network.find_imported_def("matrix_softmax_f32")
    mat_softmax.type = ThorinFnType([mem_type, passmanager_type, tensor_type], tensor_type)
    mat_sparsecrossentropy = network.find_imported_def("matrix_sparsecrossentropy_f32")
    mat_sparsecrossentropy.type = ThorinFnType([mem_type, passmanager_type, tensor_type, i32_type], f32_type)
    mat_flatten = network.find_imported_def("matrix_flatten_f32")
    mat_flatten.type = ThorinFnType([mem_type, passmanager_type, tensor_type], tensor_type)
    mat_mul = network.find_imported_def("matrix_multiply_f32")
    mat_mul.type = ThorinFnType([mem_type, passmanager_type, tensor_type, tensor_type], tensor_type)
    mat_add = network.find_imported_def("matrix_add_f32")
    mat_add.type = ThorinFnType([mem_type, passmanager_type, tensor_type, tensor_type], tensor_type)
    mat_relu = network.find_imported_def("matrix_relu_f32")
    mat_relu.type = ThorinFnType([mem_type, passmanager_type, tensor_type], tensor_type)

    with ThorinContinuation(network_exec_type, internal="run_network", thorin=network) as (run_network, run_network_mem, image, label, ret_function):
        run_network_mem, frame = thorinEnterExtract(run_network_mem)
        error_accu = ThorinSlot(frame, f32_type)

        with ThorinContinuation(body_type, filter=True) as (body_fn, body_mem, passmanager, body_return):
            allocA_continue, allocA_mem, tensorA = ThorinContinuation(tensor_return_type).__enter__()
            allocB_continue, allocB_mem, tensorB = ThorinContinuation(tensor_return_type).__enter__()
            allocAbias_continue, allocAbias_mem, tensorAbias = ThorinContinuation(tensor_return_type).__enter__()
            allocBbias_continue, allocBbias_mem, tensorBbias = ThorinContinuation(tensor_return_type).__enter__()
            allocImage_continue, allocImage_mem, tensorImage = ThorinContinuation(tensor_return_type).__enter__()

            flattenReturnCont, flatten_mem, tensorFlat = ThorinContinuation(tensor_return_type).__enter__()
            multAReturnCont, multA_mem, tensorMultA = ThorinContinuation(tensor_return_type).__enter__()
            addAReturnCont, addA_mem, tensorAddA = ThorinContinuation(tensor_return_type).__enter__()
            reluReturnCont, relu_mem, tensorRelu = ThorinContinuation(tensor_return_type).__enter__()
            multBReturnCont, multB_mem, tensorMultB = ThorinContinuation(tensor_return_type).__enter__()
            addBReturnCont, addB_mem, tensorAddB = ThorinContinuation(tensor_return_type).__enter__()
            softmaxReturnCont, softmax_mem, tensorSoftmax = ThorinContinuation(tensor_return_type).__enter__()

            with ThorinContinuation(ret_type) as (error_store_cont, return_mem, value):
                return_mem = return_mem << (error_accu, value)
                error_store_cont(body_return, return_mem)

            def rangeX_body(entry_block, mem, indexX, continueX_block):
                def rangeY_body(entry_block, mem, indexY, continueY_block):
                    image_x_ptr = ThorinLEA([image, indexX])
                    mem, image_x = mem >> image_x_ptr
                    image_x_y_ptr = ThorinLEA([image_x, indexY])

                    mem, frame = thorinEnterExtract(mem)
                    addr_ptr = ThorinSlot(frame, ThorinDefiniteArrayType(i64_type, 2))

                    addr_array = ThorinDefiniteArray(i64_type, [ThorinCast(indexX, i64_type), ThorinCast(indexY, i64_type)])
                    mem = mem << (addr_ptr, addr_array)

                    addr_ptr_opaque = ThorinBitcast(addr_ptr, ThorinPointerType(ThorinIndefiniteArrayType(i64_type)))

                    with ThorinContinuation(access_return_type) as (store_cont, store_mem, store_ptr):
                        store_mem, value = store_mem >> image_x_y_ptr
                        store_mem = store_mem << (store_ptr, value)

                        store_cont(continueY_block, store_mem)

                    image_access_fn = ThorinExtract(tensorImage, 3)
                    entry_block(image_access_fn, mem, addr_ptr_opaque, store_cont)

                def rangeY_return(return_block, mem):
                    return_block(continueX_block, mem)
                
                entry_block(*thorinRangeFn(mem, ThorinConstant(i32_type, 0), ThorinConstant(i32_type, 28), ThorinConstant(i32_type, 1), rangeY_body, rangeY_return))

            def rangeX_return(return_block, mem):
                return_block(mat_flatten, mem, passmanager, tensorImage, flattenReturnCont)
                #return_block(exec_network, mem, passmanager, label, tensorA, tensorB, tensorAbias, tensorBbias, tensorImage, error_store_cont)

            flattenReturnCont(mat_mul, flatten_mem, passmanager, tensorA, tensorFlat, multAReturnCont)
            multAReturnCont(mat_add, multA_mem, passmanager, tensorAbias, tensorMultA, addAReturnCont)
            addAReturnCont(mat_relu, addA_mem, passmanager, tensorAddA, reluReturnCont)
            reluReturnCont(mat_mul, relu_mem, passmanager, tensorB, tensorRelu, multBReturnCont)
            multBReturnCont(mat_add, multB_mem, passmanager, tensorBbias, tensorMultB, addBReturnCont)
            addBReturnCont(mat_softmax, addB_mem, passmanager, tensorAddB, softmaxReturnCont)
            softmaxReturnCont(mat_sparsecrossentropy, softmax_mem, passmanager, tensorSoftmax, label, error_store_cont)

            body_fn(*alloc_and_load_tensor(body_mem, passmanager, allocA_continue, [784, 128], "stack.0.weight"))
            allocA_continue(*alloc_and_load_tensor(allocA_mem, passmanager, allocB_continue, [128, 10], "stack.2.weight"))
            allocB_continue(*alloc_and_load_tensor(allocB_mem, passmanager, allocAbias_continue, [128], "stack.0.bias"))
            allocAbias_continue(*alloc_and_load_tensor(allocAbias_mem, passmanager, allocBbias_continue, [10], "stack.2.bias"))
            allocBbias_continue(*alloc_tensor(allocBbias_mem, passmanager, allocImage_continue, [28, 28]))
            allocImage_continue(*thorinRangeFn(allocImage_mem, ThorinConstant(i32_type, 0), ThorinConstant(i32_type, 28), ThorinConstant(i32_type, 1), rangeX_body, rangeX_return))

        with ThorinContinuation(return_type) as (return_cont, return_mem):
            return_mem, ret_value = return_mem >> error_accu
            return_cont(ret_function, return_mem, ret_value)

        with ThorinContinuation(exec_type) as (exec_cont, exec_mem, exec_int):
            exec_cont(exec_int, exec_mem, return_cont)

        run_network(sequential, run_network_mem, body_fn, exec_cont)

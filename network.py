#import onnx

from pythorin import *

with Thorin("network") as network:
    #TODO: Declare relevant functions as intern.

    network.include("network-tools.thorin.json")
    #network.import("mat.art")

    mem_type = ThorinMemType()

    f32_type = ThorinPrimType("qf32", 1)
    i32_type = ThorinPrimType("qs32", 1)
    i64_type = ThorinPrimType("qs64", 1)
    i8_type = ThorinPrimType("qs8", 1)
    u8_type = ThorinPrimType("qu8", 1)
    bool_type = ThorinPrimType("bool", 1)

    f32_ptr_type = ThorinPointerType(f32_type)

    ret_type = ThorinFnType([mem_type, f32_type])
    image_type = ThorinPointerType(ThorinIndefiniteArrayType(ThorinPointerType(ThorinIndefiniteArrayType(f32_type))))
    network_exec_type = ThorinFnType([mem_type, image_type, i32_type, ret_type])
    iarrptr_type = ThorinPointerType(ThorinIndefiniteArrayType(i8_type))
    uarrptr_type = ThorinPointerType(ThorinIndefiniteArrayType(u8_type))
    buffer_type = ThorinStructType("Buffer", [("data", iarrptr_type), ("size", i64_type), ("device", i32_type)])

    return_type = ThorinFnType([mem_type]) #8

    alloc_result_type = ThorinFnType([mem_type, buffer_type]) #47
    alloc_type = ThorinFnType([mem_type, i64_type, alloc_result_type]) #48
    release_type = ThorinFnType([mem_type, buffer_type, return_type]) #49
    passmanager_type = ThorinStructType("PassManager", [("alloc", alloc_type), ("release", release_type)])

    exec_end_type = ThorinFnType([mem_type, return_type]) #38
    exec_type = ThorinFnType([mem_type, exec_end_type]) #52
    body_type = ThorinFnType([mem_type, passmanager_type, return_type]) #51
    sequential_type = ThorinFnType([mem_type, body_type, exec_type])

    size_fn_type = ThorinFnType([mem_type, i32_type, ThorinFnType([mem_type, i64_type])])
    access_fn_type = ThorinFnType([mem_type, ThorinIndefiniteArrayType(i64_type), ThorinFnType([mem_type, ThorinPointerType(f32_type)])])
    tensor_type = ThorinStructType("Tensor_f32", [("buffer", buffer_type), ("num_dims", i32_type), ("size_fn", size_fn_type), ("access_fn", access_fn_type)]) #TODO: This is some type mangling bullshit if I have ever seen one.
    alloc_tensor_return_type = ThorinFnType([mem_type, tensor_type])

    sequential = network.find_imported_def("sequential")
    alloc_tensor = network.find_imported_def("alloc_tensor_f32")
    load_matrix = network.find_imported_def("load_matrix_into")

    #exec_network = network.find_imported_def("execute_network")

    with ThorinContinuation(network_exec_type, internal="run_network", thorin=network) as (run_network, run_network_mem, image, label, ret_function):
        run_network_mem, frame = thorinEnterExtract(run_network_mem)
        error_accu = ThorinSlot(frame, f32_type)
        run_network_mem = run_network_mem << (error_accu, ThorinConstant(f32_type, 0))

        true_const = ThorinConstant(bool_type, True)

        with ThorinContinuation(body_type) as (body_fn, body_mem, passmanager, body_return):

            ### Size functions for alloc
            with ThorinContinuation(size_fn_type) as (sizeA_lambda, size_mem, dimension, size_return):
                sizes = ThorinDefiniteArray(i64_type, [ThorinConstant(i64_type, 784), ThorinConstant(i64_type, 128)])
                size = ThorinExtract(sizes, dimension)
                sizeA_lambda(size_return, size_mem, size)

            with ThorinContinuation(size_fn_type) as (sizeB_lambda, size_mem, dimension, size_return):
                sizes = ThorinDefiniteArray(i64_type, [ThorinConstant(i64_type, 128), ThorinConstant(i64_type, 10)])
                size = ThorinExtract(sizes, dimension)
                sizeB_lambda(size_return, size_mem, size)

            with ThorinContinuation(size_fn_type) as (sizeAbias_lambda, size_mem, dimension, size_return):
                sizeAbias_lambda(size_return, size_mem, ThorinConstant(i64_type, 128))

            with ThorinContinuation(size_fn_type) as (sizeBbias_lambda, size_mem, dimension, size_return):
                sizeBbias_lambda(size_return, size_mem, ThorinConstant(i64_type, 10))

            with ThorinContinuation(size_fn_type) as (sizeImage_lambda, size_mem, dimension, size_return):
                sizeImage_lambda(size_return, size_mem, ThorinConstant(i64_type, 28))

            ### Alloc tensors
            with ThorinContinuation(alloc_tensor_return_type) as (allocA_continue, allocA_mem, tensorA):
                with ThorinContinuation(alloc_tensor_return_type) as (allocB_continue, allocB_mem, tensorB):
                    with ThorinContinuation(alloc_tensor_return_type) as (allocAbias_continue, allocAbias_mem, tensorAbias):
                        with ThorinContinuation(alloc_tensor_return_type) as (allocBbias_continue, allocBbias_mem, tensorBbias):
                            with ThorinContinuation(alloc_tensor_return_type) as (allocImage_continue, allocImage_mem, tensorImage):
                                model_name = thorinString("mnist-example/mnist.onnx")

                                #Load matricies
                                with ThorinContinuation(return_type) as (loadBbias_cont, return_mem):
                                    loadBbias_cont(body_return, return_mem)
                                with ThorinContinuation(exec_type) as (loadBbias_return, load_mem, load_int):
                                    loadBbias_return(load_int, load_mem, loadBbias_cont);

                                with ThorinContinuation(return_type) as (loadAbias_cont, return_mem):
                                    loadAbias_cont(load_matrix, return_mem, tensorBbias, model_name, thorinString("stack.2.bias"), loadBbias_return)
                                with ThorinContinuation(exec_type) as (loadAbias_return, load_mem, load_int):
                                    loadAbias_return(load_int, load_mem, loadAbias_cont);

                                with ThorinContinuation(return_type) as (loadB_cont, return_mem):
                                    loadB_cont(load_matrix, return_mem, tensorAbias, model_name, thorinString("stack.0.bias"), loadAbias_return)
                                with ThorinContinuation(exec_type) as (loadB_return, load_mem, load_int):
                                    loadB_return(load_int, load_mem, loadB_cont);

                                with ThorinContinuation(return_type) as (loadA_cont, return_mem):
                                    loadA_cont(load_matrix, return_mem, tensorB, model_name, thorinString("stack.2.weight"), loadB_return)
                                with ThorinContinuation(exec_type) as (loadA_return, load_mem, load_int):
                                    loadA_return(load_int, load_mem, loadA_cont);

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

                                        with ThorinContinuation(ThorinFnType([mem_type, f32_ptr_type])) as (store_cont, store_mem, store_ptr):
                                            store_mem, value = store_mem >> image_x_y_ptr
                                            store_mem = store_mem << (store_ptr, value)

                                            store_cont(continueY_block, store_mem)

                                        image_access_fn = ThorinExtract(tensorImage, 3)
                                        entry_block(image_access_fn, mem, addr_ptr_opaque, store_cont)

                                    def rangeY_return(return_block, mem):
                                        return_block(continueX_block, mem)
                                    
                                    entry_block(*thorinRangeFn(mem, ThorinConstant(i32_type, 0), ThorinConstant(i32_type, 28), ThorinConstant(i32_type, 1), rangeY_body, rangeY_return))

                                def rangeX_return(return_block, mem):
                                    return_block(load_matrix, mem, tensorA, model_name, thorinString("stack.0.weight"), loadA_return)

                                allocImage_continue(*thorinRangeFn(allocImage_mem, ThorinConstant(i32_type, 0), ThorinConstant(i32_type, 28), ThorinConstant(i32_type, 1), rangeX_body, rangeX_return))

                            allocBbias_continue(alloc_tensor, allocBbias_mem, passmanager, ThorinConstant(i32_type, 2), sizeImage_lambda, allocImage_continue)
                        allocAbias_continue(alloc_tensor, allocAbias_mem, passmanager, ThorinConstant(i32_type, 1), sizeBbias_lambda, allocBbias_continue)
                    allocB_continue(alloc_tensor, allocB_mem, passmanager, ThorinConstant(i32_type, 1), sizeAbias_lambda, allocAbias_continue)
                allocA_continue(alloc_tensor, allocA_mem, passmanager, ThorinConstant(i32_type, 2), sizeB_lambda, allocB_continue)
            body_fn(alloc_tensor, body_mem, passmanager, ThorinConstant(i32_type, 2), sizeA_lambda, allocA_continue)

        with ThorinContinuation(return_type) as (return_cont, return_mem):
            return_mem, ret_value = return_mem >> error_accu
            return_cont(ret_function, return_mem, ret_value)

        with ThorinContinuation(exec_type) as (exec_cont, exec_mem, exec_int):
            exec_cont(exec_int, exec_mem, return_cont)

        run_network(sequential, run_network_mem, body_fn, exec_cont)

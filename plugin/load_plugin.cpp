#include <stdio.h>
#include <iostream>

#include <thorin/world.h>
#include <thorin/analyses/cfg.h>
#include <thorin/analyses/scope.h>

#include <onnx/onnx_pb.h>
#include <onnx/proto_utils.h>
#include <onnx/common/file_utils.h>

#include <sys/resource.h>

using namespace thorin;
using namespace ONNX_NAMESPACE;

std::string const_string_parse(const Def* namedef) {
    const Bitcast * bitcast = namedef->isa<Bitcast>();
    assert(bitcast);

    const Def * bitcast_inner = bitcast->from();
    assert(bitcast_inner);

    const Global * global = bitcast_inner->isa<Global>();
    assert(global && !global->is_mutable());

    const Def * array_def = global->init();
    assert(array_def);

    const DefiniteArray * array = array_def->as<DefiniteArray>();
    assert(array);

    return array->as_string();
}

void * load_matrix (World* world, App* app) {
    const Def* matrix = app->arg(1);
    const Def* onnx_name_def = app->arg(2);
    const Def* reference_def = app->arg(3);

    const std::string model_path = const_string_parse(onnx_name_def);
    std::cerr << "Loading model " << model_path << std::endl;

    ModelProto model;
    LoadProtoFromPath(model_path, model);
    
    const std::string tensor_name = const_string_parse(reference_def);
    std::cerr << "Loading tensor for initializer " << tensor_name << std::endl;
    const TensorProto * tensor_ptr = nullptr;

    //Found the layer named by callee.
    for (int i = 0; i < model.graph().initializer_size(); i++) {
        const TensorProto &init = model.graph().initializer(i);
        if (init.name() == tensor_name) {
            std::cerr << "Found initializer " << init.name() << " as " << i << "\n";
            tensor_ptr = &init;
        } else {
            std::cerr << "Found irrelevant initializer " << init.name() << " as " << i << "\n";
        }
    }

    if (!tensor_ptr) {
        std::cerr << "Could not find the correct initializer to load.\nPossible initializers are:\n";
        for (int i = 0; i < model.graph().initializer_size(); i++) {
            const TensorProto &init = model.graph().initializer(i);
            std::cerr << "Initializer " << i << ": " << init.name() << "\n";
        }
    }
    assert(tensor_ptr);

    auto &tensor = *tensor_ptr;
    assert(tensor.data_type() == TensorProto_DataType_FLOAT);

    //Compute total number of elements to load.
    size_t tensor_num_elements = 1;
    std::cerr << tensor.dims_size() << " dims\n";
    for (int i = 0; i < tensor.dims_size(); i++) {
        std::cerr << "Dim " << i << ": " << tensor.dims(i) << "\n";
        tensor_num_elements *= tensor.dims(i);
    }
    std::cerr << "Tensor has total of " << tensor_num_elements << " elements\n";

    const std::string &data_str = tensor.raw_data();
    const char * raw_data = data_str.c_str();
    const float * data_float = (float*) raw_data;

    //Get float buffer from tensor object
    auto buffer = world->extract(matrix, (u32) 0);
    auto buffer_data = world->extract(buffer, (u32) 0);
    auto tensor_data = world->bitcast(world->ptr_type(world->indefinite_array_type(world->type_qf32())), buffer_data);

    Continuation* y = world->continuation(world->fn_type({world->mem_type(), world->fn_type({world->mem_type()})}));
    const Def* mem = y->param(0);

    //Write data directly into buffer.
    //The target data layout has to be equal to what is used in the onnx model.
    for (size_t i = 0; i < tensor_num_elements; i++) {
        auto lea = world->lea(tensor_data, world->literal_qs32(i, {}), {});
        auto data_const = world->literal_qf32(data_float[i], {});
        mem = world->store(mem, lea, data_const);
    }

    y->jump(y->param(1), {mem});

    return y;
}

extern "C" {

void * load_matrix_into (World* world, App* app) {
    //input_v[0]: file name
    //input_v[1]: matrix name
    //input_v[2]: target tensor
    //returns: void object
    //
    //TODO: how is memory handled here?
    
    return load_matrix(world, app);
}

void init_old () {
    fprintf(stdout, "Setting stack limits\n");

    struct rlimit rl;
    int result = getrlimit(RLIMIT_STACK, &rl);

    fprintf(stdout, "Current Stack limit: %d\n", rl.rlim_cur);
    rl.rlim_cur = 64L * 1024L * 1024L;
    result = setrlimit(RLIMIT_STACK, &rl);
    assert(result == 0);
}

}

void * static_memory (World* world, App* app) {
    const Def* size = app->arg(1);

    Continuation* y = world->continuation(world->fn_type({
                world->mem_type(),
                world->fn_type({
                        world->mem_type(),
                        world->ptr_type(world->indefinite_array_type(world->type_qs8()))
                        })
                }));

    const Def* mem = y->param(0);
    const Def* array;

    if (auto size_lit = size->isa<PrimLit>()) {
        u64 size = size_lit->value().get_u64();
        //std::cerr << "Allocating memory of size " << size << "\n";
        auto target_type = world->ptr_type(world->indefinite_array_type(world->type_qs8()));
        const Def* inner_array = world->global(world->bottom(world->definite_array_type(world->type_qs8(), size)));
        array = world->bitcast(target_type, inner_array);
    } else {
        //const Def* inner_alloca = world->alloc(world->indefinite_array_type(world->type_qs8()), mem, size);
        //mem = world->extract(inner_alloca, (int) 0);
        //array = world->extract(inner_alloca, 1);
        array = world->bitcast(world->ptr_type(world->indefinite_array_type(world->type_qs8())), world->literal_qu64(0, {}));
    }

    y->jump(y->param(1), {mem, array});

    return y;
}

void * static_free (World* world, App* app) {
    const Def* data = app->arg(1);

    Continuation* y = world->continuation(world->fn_type({
                world->mem_type(),
                world->fn_type({
                        world->mem_type()
                        })
                }));

    const Def* mem = y->param(0);
    y->jump(y->param(1), {mem});

    return y;
}

void * build_static_array_cpp (World* world, App* app) {
    //std::cerr << "build_static_array\n";
    //app->dump();
    const Def* element = app->arg(1);
    const Def* size = app->arg(2);

    auto size_lit = size->isa<PrimLit>();
    //assert(size_lit && "Static arrays need to have a known size at this point.");
    if (!size_lit) {
        throw std::runtime_error("Static arrays need to have a known size at this point.");
    }

    u64 array_size = size_lit->value().get_u64();

    Array<const Def*> elems(array_size);

    //std::cerr << "Array Size " << array_size << "\n";

    for (u64 i = 0; i < array_size; i++) {
        elems[i] = element;
    }

    const Def* plain_array = world->definite_array(element->type(), elems);
    const Def* global_array = world->global(plain_array, false);

    const Def* result = world->bitcast(world->ptr_type(world->indefinite_array_type(element->type())), global_array);

    return const_cast<Def*>(result);
}

void * build_dynamic_array_cpp (World* world, App* app) {
    //std::cerr << "build_static_array\n";
    //app->dump();
    //app->callee()->type()->dump();
    
    const Def* element = app->arg(1);
    const Def* size = app->arg(2);

    auto size_lit = size->isa<PrimLit>();
    //assert(size_lit && "Static arrays need to have a known size at this point.");
    if (!size_lit) {
        throw std::runtime_error("Dynamic arrays need to have a known size at this point.");
    }

    u64 array_size = size_lit->value().get_u64();

    Array<const Def*> elems(array_size);

    //std::cerr << "Array Size " << array_size << "\n";

    //for (u64 i = 0; i < array_size; i++) {
    //    elems[i] = element;
    //}

    //const Def* plain_array = world->definite_array(element->type(), elems);

    //const Def* result = world->bitcast(world->ptr_type(world->indefinite_array_type(element->type())), global_array);
    
    auto array_type = world->definite_array_type(element->type(), array_size);
    auto array_return_type = world->ptr_type(world->indefinite_array_type(element->type()));
    
    Continuation* y = world->continuation(world->fn_type({world->mem_type(),
                                                          world->fn_type({world->mem_type(),
                                                                          array_return_type
                                                                         })
                                                         })
            );

    const Def* mem = y->param(0);

    auto pair = world->enter(mem);
    mem = world->extract(pair, thorin::u32(0));
    auto frame = world->extract(pair, thorin::u32(1));

    auto slot = world->slot(array_type, frame);
    auto array = world->bitcast(array_return_type, slot);

    y->jump(y->param(1), {mem, array});

    //return const_cast<Def*>(result);
    //return nullptr;
    return y;
}

void * static_array_set_element_cpp (World* world, App* app) {
    //std::cerr << "static_array_set_element\n";
    //app->dump();
    const Def* element = app->arg(1);
    const Def* value = app->arg(2);

    auto lea = element->as<LEA>();

    auto index_lit = lea->index()->isa<PrimLit>();
    assert(index_lit);
    u64 index = index_lit->value().get_u64();

    auto global_bitcast = lea->ptr()->isa<Bitcast>();
    assert(global_bitcast);

    auto global = global_bitcast->from()->isa<Global>();
    assert(global);
    auto array = global->init()->isa<DefiniteArray>();
    assert(array);

    auto element_type = array->type()->elem_type();

    //array->dump();

    size_t array_size = array->num_ops();

    //std::cerr << "Array Size " << array_size << "\n";
    //std::cerr << "Index " << index << "\n";

    Array<const Def*> elems(array_size);
    for (u64 i = 0; i < array_size; i++) {
        elems[i] = array->op(i);
    }

    elems[index] = value;

    const Def* plain_array = world->definite_array(element_type, elems);
    array->replace_uses(plain_array);

    //std::cerr << "Setting element at index " << index << " with:\n";
    //value->dump();

    return nullptr;
}

extern "C" {

void * static_alloca (World* world, App* app) {
    return static_memory(world, app);
}

void * static_release (World* world, App* app) {
    return static_free(world, app);
}

void * build_static_array (World* world, App* app) {
    return build_static_array_cpp(world, app);
}

void * build_dynamic_array (World* world, App* app) {
    return build_dynamic_array_cpp(world, app);
}

void * static_array_set_element (World* world, App* app) {
    return static_array_set_element_cpp(world, app);
}

}

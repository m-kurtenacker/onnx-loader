#include <stdio.h>

#include <thorin/world.h>
#include <thorin/analyses/cfg.h>
#include <thorin/analyses/scope.h>

#include <onnx/onnx_pb.h>
#include <onnx/proto_utils.h>
#include <onnx/common/file_utils.h>

#include <sys/resource.h>
#include <stdio.h>

using namespace thorin;
using namespace ONNX_NAMESPACE;

std::string const_string_parse(Def* namedef) {
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

void * test (void * somedef1, void * somedef2, void * somedef3) {
    Def* matrix = (Def *) somedef1;
    Def* onnx_name_def = (Def*) somedef2;
    Def* reference_def = (Def*) somedef3;

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

    World& world = matrix->world();

    //Get float buffer from tensor object
    auto buffer = world.extract(matrix, (u32) 0);
    auto buffer_data = world.extract(buffer, (u32) 0);
    auto tensor_data = world.bitcast(world.ptr_type(world.indefinite_array_type(world.type_qf32())), buffer_data);

    Continuation* y = world.continuation(world.fn_type({world.mem_type(), world.fn_type({world.mem_type()})}));
    const Def* mem = y->param(0);

    //Write data directly into buffer.
    //The target data layout has to be equal to what is used in the onnx model.
    for (size_t i = 0; i < tensor_num_elements; i++) {
        auto lea = world.lea(tensor_data, world.literal_qs32(i, {}), {});
        auto data_const = world.literal_qf32(data_float[i], {});
        mem = world.store(mem, lea, data_const);
    }

    y->jump(y->param(1), {mem});

    return y;
}

extern "C" {

void * load_matrix_into (size_t input_c, void ** input_v) {
    //input_v[0]: file name
    //input_v[1]: matrix name
    //input_v[2]: target tensor
    //returns: void object
    //
    //TODO: how is memory handled here?
    
    assert(input_c == 3);
    return test(input_v[0], input_v[1], input_v[2]);
}

}

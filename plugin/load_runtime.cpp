#include <stdio.h>

#include <onnx/onnx_pb.h>
#include <onnx/proto_utils.h>
#include <onnx/common/file_utils.h>

using namespace ONNX_NAMESPACE;

void load_matrix_dynamic_cpp (float * tensor_buffer, char * file_name, char * matrix_name) {
    static int i;
    if (i >= 4)
        return;
    i++;

    const std::string model_path(file_name);
    //std::cerr << "Loading model " << model_path << std::endl;

    ModelProto model;
    LoadProtoFromPath(model_path, model);
    
    const std::string tensor_name(matrix_name);
    //std::cerr << "Loading tensor for initializer " << tensor_name << std::endl;
    const TensorProto * tensor_ptr = nullptr;

    //Found the layer named by callee.
    for (int i = 0; i < model.graph().initializer_size(); i++) {
        const TensorProto &init = model.graph().initializer(i);
        if (init.name() == tensor_name) {
            //std::cerr << "Found initializer " << init.name() << " as " << i << "\n";
            tensor_ptr = &init;
        } else {
            //std::cerr << "Found irrelevant initializer " << init.name() << " as " << i << "\n";
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
    //std::cerr << tensor.dims_size() << " dims\n";
    for (int i = 0; i < tensor.dims_size(); i++) {
        //std::cerr << "Dim " << i << ": " << tensor.dims(i) << "\n";
        tensor_num_elements *= tensor.dims(i);
    }
    //std::cerr << "Tensor has total of " << tensor_num_elements << " elements\n";

    const std::string &data_str = tensor.raw_data();
    const char * raw_data = data_str.c_str();
    const float * data_float = (float*) raw_data;

    for (size_t i = 0; i < tensor_num_elements; i++) {
        tensor_buffer[i] = data_float[i];
    }
}

extern "C" {

void load_matrix_dynamic(float * tensor, char * file_name, char * matrix_name) {
    load_matrix_dynamic_cpp(tensor, file_name, matrix_name);
}

}

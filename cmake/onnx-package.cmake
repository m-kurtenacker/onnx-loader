set(VENV "${CMAKE_CURRENT_BINARY_DIR}/venv")

include(FetchContent)

FetchContent_Declare(onnx
    GIT_REPOSITORY https://github.com/onnx/onnx
    GIT_TAG origin/main
    FIND_PACKAGE_ARGS
)
message(STATUS "Making ONNX available...")
FetchContent_MakeAvailable(onnx)
message(STATUS "ONNX source folder is ${onnx_SOURCE_DIR}")

find_package(Python REQUIRED COMPONENTS Interpreter)
add_custom_command(
    OUTPUT ${VENV}
    COMMAND ${Python_EXECUTABLE} -m venv ${VENV}
    COMMAND ${VENV}/bin/pip install numpy idx2numpy
    COMMAND ${VENV}/bin/pip install -e ${onnx_SOURCE_DIR}/
)

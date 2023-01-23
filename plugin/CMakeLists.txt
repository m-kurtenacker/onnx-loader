cmake_minimum_required(VERSION 3.23.3 FATAL_ERROR)

project (onnx_loader)

option(PROFILE "profile the loader execution with gprof" OFF)

find_package(Thorin REQUIRED)
find_package(Protobuf REQUIRED)
find_package(ONNX REQUIRED)

add_library(
    loader
    SHARED
    load.cpp
    memory.cpp
)
set_target_properties(loader PROPERTIES PREFIX "" CXX_STANDARD 17)

target_link_libraries(loader ${Thorin_LIBRARIES} onnx)

set(CMAKE_CXX_FLAGS "-DONNX_ML=1 -DONNX_NAMESPACE=onnx")

if (PROFILE)
    message("-- Building with profiling support")
    set(CMAKE_CXX_FLAGS "-pg")
    set(CMAKE_SHARED_LINKER_FLAGS "-pg")
endif()

target_include_directories(loader PUBLIC ${Thorin_INCLUDE_DIRS} ${ONNX_INCLUDE_DIRS})

# Generate test data on the fly
add_custom_command(OUTPUT test.idx
    COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/generate_data.py
)

add_custom_command(OUTPUT test.onnx
    COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/generate_model.py
)


# Build model
add_custom_command(
    OUTPUT network.thorin.json
    COMMAND ${VENV}/bin/python ${PROJECT_SOURCE_DIR}/network.py ${CMAKE_CURRENT_BINARY_DIR}/test.onnx ${NET_TOOLS_JSON}
    DEPENDS ${PROJECT_SOURCE_DIR}/network.py ${NET_TOOLS_JSON} ${VENV} ${CMAKE_CURRENT_BINARY_DIR}/test.onnx
)

anydsl_runtime_fancy_wrap(RESHAPE_NETWORK_WITHTOOLS_JSON
    FRONTEND "anyopt"
    ANYOPT_FLAGS --pass cleanup_fix_point --keep-intern run_network
    EMIT_JSON
    NAME "network_withtools"
    FILES ${CMAKE_CURRENT_BINARY_DIR}/network.thorin.json ${NET_TOOLS_JSON}
)

anydsl_runtime_fancy_wrap(RESHAPE_NETWORK_COMPILED_JSON
    FRONTEND "anyopt"
    ANYOPT_FLAGS --pass cleanup --pass lower2cff --pass cleanup
    EMIT_JSON
    NAME "network_compiled"
    PLUGINS plugin/load_plugin.cpp
    FILES ${RESHAPE_NETWORK_WITHTOOLS_JSON}
)


# Build main + link everyting
anydsl_runtime_fancy_wrap(TEST_RESHAPE_MAIN_JSON
    FRONTEND "artic"
    NAME "test"
    EMIT_JSON
    FILES test.art ${PROJECT_SOURCE_DIR}/sequential.art ${PROJECT_SOURCE_DIR}/mat.art ${PROJECT_SOURCE_DIR}/utils/read.art ${MAT_OPS}
)

anydsl_runtime_fancy_wrap(TEST_RESHAPE_MAIN_INT
    FRONTEND "anyopt"
    ANYOPT_FLAGS --pass cleanup --pass lower2cff --pass cleanup --remove-interns
    EMIT_JSON
    NAME "test-compiled"
    FILES ${TEST_RESHAPE_MAIN_JSON}
)

anydsl_runtime_fancy_wrap(TEST_RESHAPE_ONNX_OBJ
    FRONTEND "anyopt"
    PLUGINS plugin/load_plugin.cpp
    FILES ${TEST_RESHAPE_MAIN_INT} ${RESHAPE_NETWORK_COMPILED_JSON}
)

add_executable(test_reshape_onnx ${TEST_RESHAPE_ONNX_OBJ} ${PROJECT_SOURCE_DIR}/utils/read.cpp ${PROJECT_SOURCE_DIR}/utils/allocator.cpp)
target_link_libraries(test_reshape_onnx ${AnyDSL_runtime_LIBRARIES})
set_property(TARGET test_reshape_onnx APPEND PROPERTY OBJECT_DEPENDS test.idx)

add_test(NAME reshape_onnx COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/test.py ${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

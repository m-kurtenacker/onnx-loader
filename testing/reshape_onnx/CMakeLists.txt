add_custom_command(OUTPUT test.onnx
    COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/generate_model.py
    DEPENDS generate_model.py ${VENV}
)


# Build model
add_custom_command(
    OUTPUT network.thorin.json
    COMMAND ${VENV}/bin/python ${PROJECT_SOURCE_DIR}/lib/network.py ${CMAKE_CURRENT_BINARY_DIR}/test.onnx ${NET_TOOLS_JSON}
    DEPENDS ${PROJECT_SOURCE_DIR}/lib/network.py ${NET_TOOLS_JSON} ${VENV} ${CMAKE_CURRENT_BINARY_DIR}/test.onnx
)

anydsl_runtime_wrap(RESHAPE_NETWORK_WITHTOOLS_JSON
    FRONTEND "anyopt"
    ANYOPT_FLAGS --keep-intern setup_network --pass cleanup_fix_point
    EMIT_JSON
    NAME "network_withtools"
    FILES ${CMAKE_CURRENT_BINARY_DIR}/network.thorin.json ${NET_TOOLS_JSON}
)


# Build main + link everyting
anydsl_runtime_wrap(TEST_RESHAPE_MAIN_JSON
    FRONTEND "artic"
    NAME "test"
    EMIT_JSON
    FILES test.art ${SUPPORT_FILES}
)

anydsl_runtime_wrap(TEST_RESHAPE_MAIN_INT
    FRONTEND "anyopt"
    ANYOPT_FLAGS --keep-intern setup_network --pass cleanup_fix_point
    EMIT_JSON
    NAME "test-compiled"
    FILES ${TEST_RESHAPE_MAIN_JSON}
)

anydsl_runtime_wrap(TEST_RESHAPE_ONNX_COMBINED
    FRONTEND "anyopt"
    ANYOPT_FLAGS --remove-interns
    EMIT_JSON
    NAME "test-combined"
    FILES ${TEST_RESHAPE_MAIN_INT} ${RESHAPE_NETWORK_WITHTOOLS_JSON}
)

anydsl_runtime_wrap(TEST_RESHAPE_ONNX_OBJ
    FRONTEND "anyopt"
    PLUGINS plugin/load_plugin.cpp
    FILES ${TEST_RESHAPE_ONNX_COMBINED}
)

add_executable(test_reshape_onnx ${TEST_RESHAPE_ONNX_OBJ} ${PROJECT_SOURCE_DIR}/src/utils/read.cpp ${PROJECT_SOURCE_DIR}/src/utils/allocator.cpp)
target_link_libraries(test_reshape_onnx loader_runtime ${AnyDSL_runtime_LIBRARIES})

add_test(NAME reshape_onnx COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/test.py ${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})


# Generate test data on the fly
add_custom_target(RESHAPE_ONNX_TEST_IDX
    COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/generate_data.py
    DEPENDS generate_data.py ${VENV}
)

add_test(build_reshape_onnx_testidx
  "${CMAKE_COMMAND}"
  --build "${CMAKE_BINARY_DIR}"
  --config "$<CONFIG>"
  --target RESHAPE_ONNX_TEST_IDX
)

set_tests_properties(build_reshape_onnx_testidx PROPERTIES FIXTURES_SETUP reshape_onnx_fixture)
set_tests_properties(reshape_onnx PROPERTIES FIXTURES_REQUIRED reshape_onnx_fixture)

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

anydsl_runtime_wrap(CONV_NETWORK_WITHTOOLS_JSON
    FRONTEND "anyopt"
    ANYOPT_FLAGS --pass cleanup_fix_point --keep-intern setup_network
    EMIT_JSON
    NAME "network_withtools"
    FILES ${CMAKE_CURRENT_BINARY_DIR}/network.thorin.json ${NET_TOOLS_JSON}
)

anydsl_runtime_wrap(CONV_NETWORK_COMPILED_JSON
    FRONTEND "anyopt"
    ANYOPT_FLAGS --pass cleanup --pass lower2cff --pass cleanup
    EMIT_JSON
    NAME "network_compiled"
    PLUGINS plugin/load_plugin.cpp
    FILES ${CONV_NETWORK_WITHTOOLS_JSON}
)


# Build main + link everyting
anydsl_runtime_wrap(TEST_CONV_MAIN_JSON
    FRONTEND "artic"
    NAME "test"
    EMIT_JSON
    FILES test.art ${SUPPORT_FILES}
)

anydsl_runtime_wrap(TEST_CONV_MAIN_INT
    FRONTEND "anyopt"
    ANYOPT_FLAGS --pass cleanup --pass lower2cff --pass cleanup --keep-intern setup_network
    EMIT_JSON
    NAME "test-compiled"
    FILES ${TEST_CONV_MAIN_JSON}
)

#TODO: See openproject bug#631 for why this step is needed.
anydsl_runtime_wrap(TEST_CONV_ONNX_COMBINED
    FRONTEND "anyopt"
    ANYOPT_FLAGS --pass cleanup --remove-interns
    EMIT_JSON
    NAME "test-combined"
    PLUGINS plugin/load_plugin.cpp
    FILES ${TEST_CONV_MAIN_INT} ${CONV_NETWORK_COMPILED_JSON}
)

anydsl_runtime_wrap(TEST_CONV_ONNX_OBJ
    FRONTEND "anyopt"
    FILES ${TEST_CONV_ONNX_COMBINED}
)

add_executable(test_conv_onnx ${TEST_CONV_ONNX_OBJ} ${PROJECT_SOURCE_DIR}/src/utils/read.cpp ${PROJECT_SOURCE_DIR}/src/utils/allocator.cpp)
target_link_libraries(test_conv_onnx loader_runtime ${AnyDSL_runtime_LIBRARIES})

add_test(NAME conv_onnx COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/test.py ${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})


# Generate test data on the fly
add_custom_target(CONV_ONNX_TEST_IDX
    COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/generate_data.py
    DEPENDS generate_data.py ${VENV}
)

add_test(build_conv_onnx_testidx
  "${CMAKE_COMMAND}"
  --build "${CMAKE_BINARY_DIR}"
  --config "$<CONFIG>"
  --target CONV_ONNX_TEST_IDX
)

set_tests_properties(build_conv_onnx_testidx PROPERTIES FIXTURES_SETUP conv_onnx_fixture)
set_tests_properties(conv_onnx PROPERTIES FIXTURES_REQUIRED conv_onnx_fixture)

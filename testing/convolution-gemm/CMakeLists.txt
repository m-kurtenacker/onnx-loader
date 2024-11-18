# Build main + link everyting
anydsl_runtime_wrap(TEST_CONV_GEMM_MAIN_JSON
    FRONTEND "artic"
    NAME "test"
    EMIT_JSON
    FILES test.art ${SUPPORT_FILES}
)

anydsl_runtime_wrap(TEST_CONV_GEMM_MAIN_INT
    FRONTEND "anyopt"
    ANYOPT_FLAGS --pass cleanup --pass lower2cff --pass cleanup --remove-interns
    EMIT_JSON
    NAME "test-compiled"
    FILES ${TEST_CONV_GEMM_MAIN_JSON}
)

anydsl_runtime_wrap(TEST_CONV_GEMM_OBJ
    FRONTEND "anyopt"
    PLUGINS plugin/load_plugin.cpp
    FILES ${TEST_CONV_GEMM_MAIN_INT}
)

add_executable(test_conv_gemm ${TEST_CONV_GEMM_OBJ} ${PROJECT_SOURCE_DIR}/src/utils/read.cpp ${PROJECT_SOURCE_DIR}/src/utils/allocator.cpp)
target_link_libraries(test_conv_gemm loader_runtime ${AnyDSL_runtime_LIBRARIES})

add_test(NAME conv_gemm COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/test.py ${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})


# Generate test data on the fly
add_custom_target(CONV_GEMM_TEST_IDX
    COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/generate_data.py
    DEPENDS generate_data.py ${VENV}
)

add_custom_target(CONV_GEMM_TEST_ONNX
    COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/generate_model.py
    DEPENDS generate_model.py ${VENV}
)

add_test(build_conv_gemm_testidx
  "${CMAKE_COMMAND}"
  --build "${CMAKE_BINARY_DIR}"
  --config "$<CONFIG>"
  --target CONV_GEMM_TEST_IDX
)

add_test(build_conv_gemm_testonnx
  "${CMAKE_COMMAND}"
  --build "${CMAKE_BINARY_DIR}"
  --config "$<CONFIG>"
  --target CONV_GEMM_TEST_ONNX
)

set_tests_properties(build_conv_gemm_testidx PROPERTIES FIXTURES_SETUP conv_gemm_fixture)
set_tests_properties(build_conv_gemm_testonnx PROPERTIES FIXTURES_SETUP conv_gemm_fixture)
set_tests_properties(conv_gemm PROPERTIES FIXTURES_REQUIRED conv_gemm_fixture)

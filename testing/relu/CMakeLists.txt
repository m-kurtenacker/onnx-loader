anydsl_runtime_wrap(TEST_RELU_JSON
    FRONTEND "artic"
    NAME "test"
    EMIT_JSON
    FILES test.art ${SUPPORT_FILES}
)

anydsl_runtime_wrap(TEST_REMOVE_INT
    FRONTEND "anyopt"
    ANYOPT_FLAGS "--remove-interns"
    EMIT_JSON
    NAME "test-2"
    FILES ${TEST_RELU_JSON}
)

anydsl_runtime_wrap(TEST_RELU_OBJ
    FRONTEND "anyopt"
    PLUGINS plugin/load_plugin.cpp
    FILES ${TEST_REMOVE_INT}
)

add_executable(test_relu ${TEST_RELU_OBJ} ${PROJECT_SOURCE_DIR}/src/utils/read.cpp ${PROJECT_SOURCE_DIR}/src/utils/allocator.cpp)
target_link_libraries(test_relu ${AnyDSL_runtime_LIBRARIES})

add_test(NAME relu_simple COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/test.py ${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Generate test data on the fly
add_custom_target(RELU_TEST_IDX
    COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/generate_data.py
    DEPENDS generate_data.py ${VENV}
)

add_test(build_relu_testidx
  "${CMAKE_COMMAND}"
  --build "${CMAKE_BINARY_DIR}"
  --config "$<CONFIG>"
  --target RELU_TEST_IDX
)

set_tests_properties(build_relu_testidx PROPERTIES FIXTURES_SETUP relu_fixture)
set_tests_properties(relu_simple PROPERTIES FIXTURES_REQUIRED relu_fixture)

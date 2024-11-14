anydsl_runtime_wrap(TEST_IM2COL_JSON
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
    FILES ${TEST_IM2COL_JSON}
)

anydsl_runtime_wrap(TEST_IM2COL_OBJ
    FRONTEND "anyopt"
    PLUGINS plugin/load_plugin.cpp
    FILES ${TEST_REMOVE_INT}
)

add_executable(test_im2col ${TEST_IM2COL_OBJ} ${PROJECT_SOURCE_DIR}/src/utils/read.cpp ${PROJECT_SOURCE_DIR}/src/utils/allocator.cpp)
target_link_libraries(test_im2col ${AnyDSL_runtime_LIBRARIES})

add_test(NAME im2col_simple COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/test.py ${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})


# Generate test data on the fly
add_custom_target(IM2COL_TEST_IDX
    COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/generate_data.py
    DEPENDS generate_data.py ${VENV}
)

add_test(build_im2col_testidx
  "${CMAKE_COMMAND}"
  --build "${CMAKE_BINARY_DIR}"
  --config "$<CONFIG>"
  --target IM2COL_TEST_IDX
)

set_tests_properties(build_im2col_testidx PROPERTIES FIXTURES_SETUP im2col_fixture)
set_tests_properties(im2col_simple PROPERTIES FIXTURES_REQUIRED im2col_fixture)

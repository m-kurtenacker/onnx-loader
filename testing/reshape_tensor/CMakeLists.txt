anydsl_runtime_fancy_wrap(TEST_RESHAPE_TENSOR_JSON
    FRONTEND "artic"
    NAME "test"
    EMIT_JSON
    FILES test.art ${SUPPORT_FILES}
)

anydsl_runtime_fancy_wrap(TEST_REMOVE_INT_TENSOR
    FRONTEND "anyopt"
    ANYOPT_FLAGS "--remove-interns"
    EMIT_JSON
    NAME "test-2"
    FILES ${TEST_RESHAPE_TENSOR_JSON}
)

anydsl_runtime_fancy_wrap(TEST_RESHAPE_TENSOR_OBJ
    FRONTEND "anyopt"
    PLUGINS plugin/load_plugin.cpp
    FILES ${TEST_REMOVE_INT_TENSOR}
)

add_executable(test_reshape_tensor ${TEST_RESHAPE_TENSOR_OBJ} ${PROJECT_SOURCE_DIR}/src/utils/read.cpp ${PROJECT_SOURCE_DIR}/src/utils/allocator.cpp)
target_link_libraries(test_reshape_tensor ${AnyDSL_runtime_LIBRARIES})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test.idx DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME reshape_tensor COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/test.py ${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

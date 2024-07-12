anydsl_runtime_fancy_wrap(TEST_RESHAPE_JSON
    FRONTEND "artic"
    NAME "test"
    EMIT_JSON
    FILES test.art ${PROJECT_SOURCE_DIR}/sequential.art ${PROJECT_SOURCE_DIR}/mat.art ${PROJECT_SOURCE_DIR}/utils/read.art ${MAT_OPS}
    #FILES test.art ${PROJECT_SOURCE_DIR}/mat.art ${PROJECT_SOURCE_DIR}/sequential.art ${PROJECT_SOURCE_DIR}/utils/read.art ${PROJECT_SOURCE_DIR}/utils/utils.art
)

anydsl_runtime_fancy_wrap(TEST_REMOVE_INT
    FRONTEND "anyopt"
    ANYOPT_FLAGS "--remove-interns"
    EMIT_JSON
    NAME "test-2"
    FILES ${TEST_RESHAPE_JSON}
)

anydsl_runtime_fancy_wrap(TEST_RESHAPE_OBJ
    FRONTEND "anyopt"
    PLUGINS plugin/load_plugin.cpp
    FILES ${TEST_REMOVE_INT}
)

add_executable(test_reshape ${TEST_RESHAPE_OBJ} ${PROJECT_SOURCE_DIR}/utils/read.cpp ${PROJECT_SOURCE_DIR}/utils/allocator.cpp)
target_link_libraries(test_reshape ${AnyDSL_runtime_LIBRARIES})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test.idx DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME reshape_simple COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/test.py ${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

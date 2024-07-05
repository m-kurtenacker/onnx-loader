anydsl_runtime_fancy_wrap(TEST_ADD_JSON
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
    FILES ${TEST_ADD_JSON}
)

anydsl_runtime_fancy_wrap(TEST_ADD_OBJ
    FRONTEND "anyopt"
    PLUGINS plugin/load_plugin.cpp
    FILES ${TEST_REMOVE_INT}
)

add_executable(test_add ${TEST_ADD_OBJ} ${PROJECT_SOURCE_DIR}/utils/read.cpp ${PROJECT_SOURCE_DIR}/utils/allocator.cpp)
target_link_libraries(test_add ${AnyDSL_runtime_LIBRARIES})

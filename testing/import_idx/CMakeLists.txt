add_custom_command(
    OUTPUT load_test.thorin.json
    COMMAND PYTHONPATH=${PROJECT_SOURCE_DIR}/lib ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/generate_data.py ${NET_TOOLS_JSON}
    DEPENDS generate_data.py ${NET_TOOLS_JSON} ${VENV}
)

# Build main + link everyting
anydsl_runtime_wrap(TEST_LOAD_IDX_MAIN_JSON
    FRONTEND "artic"
    NAME "main"
    EMIT_JSON
    FILES main.art ${SUPPORT_FILES}
)

anydsl_runtime_wrap(TEST_LOAD_IDX_COMBINED_JSON
    FRONTEND "anyopt"
    ANYOPT_FLAGS --pass cleanup_fix_point --remove-interns
    NAME "combined"
    EMIT_JSON
    FILES ${TEST_LOAD_IDX_MAIN_JSON} ${CMAKE_CURRENT_BINARY_DIR}/load_test.thorin.json
)

anydsl_runtime_wrap(TEST_LOAD_IDX_OBJ
    FRONTEND "anyopt"
    PLUGINS plugin/load_plugin.cpp
    FILES ${TEST_LOAD_IDX_COMBINED_JSON}
)

add_executable(test_load_idx ${TEST_LOAD_IDX_OBJ} ${PROJECT_SOURCE_DIR}/src/utils/read.cpp ${PROJECT_SOURCE_DIR}/src/utils/allocator.cpp)
target_link_libraries(test_load_idx loader_runtime ${AnyDSL_runtime_LIBRARIES})

add_test(NAME transform_idx COMMAND test_load_idx WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

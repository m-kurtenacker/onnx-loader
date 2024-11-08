# Generate test data on the fly
add_custom_command(OUTPUT loader.thorin.json
    COMMAND PYTHONPATH=${PROJECT_SOURCE_DIR}/lib ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/generate_data.py ${NET_TOOLS_JSON}
    BYPRODUCTS A.idx B.idx C.idx
    DEPENDS generate_data.py ${NET_TOOLS_JSON} ${VENV}
)


# Build main + link everyting
anydsl_runtime_wrap(TEST_GEMM_MAIN_JSON
    FRONTEND "artic"
    NAME "test"
    EMIT_JSON
    FILES test.art ${SUPPORT_FILES}
)

anydsl_runtime_wrap(TEST_GEMM_MAIN_COMBINED
    FRONTEND "anyopt"
    ANYOPT_FLAGS --pass cleanup_fix_point --remove-interns
    NAME "combined"
    EMIT_JSON
    FILES ${TEST_GEMM_MAIN_JSON} ${CMAKE_CURRENT_BINARY_DIR}/loader.thorin.json
)

anydsl_runtime_wrap(TEST_GEMM_OBJ
    FRONTEND "anyopt"
    PLUGINS plugin/load_plugin.cpp
    FILES ${TEST_GEMM_MAIN_COMBINED}
)

add_executable(test_gemm ${TEST_GEMM_OBJ} ${PROJECT_SOURCE_DIR}/src/utils/read.cpp ${PROJECT_SOURCE_DIR}/src/utils/allocator.cpp)
target_link_libraries(test_gemm loader_runtime ${AnyDSL_runtime_LIBRARIES})

add_test(NAME test_gemm COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/test.py ${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

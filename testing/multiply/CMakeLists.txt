# Generate test data on the fly
add_custom_command(OUTPUT loader.thorin.json
    COMMAND PYTHONPATH=${PROJECT_SOURCE_DIR}/lib ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/generate_data.py ${NET_TOOLS_JSON}
    BYPRODUCTS M.idx t.idx
    DEPENDS generate_data.py ${VENV}
)


# Link loader to network tools
anydsl_runtime_wrap(LOADER_WITHTOOLS_JSON
    FRONTEND "anyopt"
    ANYOPT_FLAGS --pass cleanup_fix_point --keep-intern load_M --keep-intern load_t
    EMIT_JSON
    NAME "loader_withtools"
    FILES ${CMAKE_CURRENT_BINARY_DIR}/loader.thorin.json ${NET_TOOLS_JSON}
)

#anydsl_runtime_wrapL(LOADER_COMPILED_JSON
#FRONTEND "anyopt"
#ANYOPT_FLAGS --pass cleanup --pass lower2cff --pass cleanup
#EMIT_JSON
#NAME "loader_compiled"
#PLUGINS plugin/load_plugin.cpp
#FILES ${LOADER_WITHTOOLS_JSON}
#)

# Build main + link everyting
anydsl_runtime_wrap(TEST_MULT_MAIN_JSON
    FRONTEND "artic"
    NAME "test"
    EMIT_JSON
    FILES test.art ${SUPPORT_FILES}
)

anydsl_runtime_wrap(TEST_MULT_MAIN_INT
    FRONTEND "anyopt"
    ANYOPT_FLAGS --pass cleanup --pass lower2cff --pass cleanup --keep-intern load_M --keep-intern load_t
    EMIT_JSON
    NAME "test-compiled"
    FILES ${TEST_MULT_MAIN_JSON}
)

anydsl_runtime_wrap(TEST_MULT_OBJ
    FRONTEND "anyopt"
    PLUGINS plugin/load_plugin.cpp
    FILES ${TEST_MULT_MAIN_INT} ${LOADER_WITHTOOLS_JSON}
)

add_executable(test_multiply ${TEST_MULT_OBJ} ${PROJECT_SOURCE_DIR}/src/utils/read.cpp ${PROJECT_SOURCE_DIR}/src/utils/allocator.cpp)
target_link_libraries(test_multiply loader_runtime ${AnyDSL_runtime_LIBRARIES})

add_test(NAME test_multiply COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/test.py ${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

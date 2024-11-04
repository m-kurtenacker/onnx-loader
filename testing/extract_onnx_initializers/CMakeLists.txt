# Generate test data on the fly
add_custom_command(OUTPUT test.onnx
    COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/generate_model.py
    DEPENDS generate_model.py ${VENV}
)

add_custom_command(
    OUTPUT loader.thorin.json
    COMMAND PYTHONPATH=${PROJECT_SOURCE_DIR}/lib ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/network.py ${CMAKE_CURRENT_BINARY_DIR}/test.onnx ${NET_TOOLS_JSON}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/network.py ${NET_TOOLS_JSON} ${VENV} ${CMAKE_CURRENT_BINARY_DIR}/test.onnx
)

anydsl_runtime_wrap(TEST_EXPORT_IDX_MAIN_JSON
    FRONTEND "artic"
    NAME "main"
    EMIT_JSON
    FILES main.art ${SUPPORT_FILES}
)

anydsl_runtime_wrap(TEST_EXPORT_IDX_COMBINED_JSON
    FRONTEND "anyopt"
    ANYOPT_FLAGS --pass cleanup_fix_point --remove-interns
    NAME "combined"
    EMIT_JSON
    FILES ${TEST_EXPORT_IDX_MAIN_JSON} ${CMAKE_CURRENT_BINARY_DIR}/loader.thorin.json
)

anydsl_runtime_wrap(TEST_EXPORT_IDX_OBJ
    FRONTEND "anyopt"
    PLUGINS plugin/load_plugin.cpp
    FILES ${TEST_EXPORT_IDX_COMBINED_JSON}
)

add_executable(test_export_idx ${TEST_EXPORT_IDX_OBJ} ${PROJECT_SOURCE_DIR}/src/utils/read.cpp ${PROJECT_SOURCE_DIR}/src/utils/allocator.cpp)
target_link_libraries(test_export_idx loader_runtime ${AnyDSL_runtime_LIBRARIES})

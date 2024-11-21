# Build main + link everyting
anydsl_runtime_wrap(TEST_CONV_SUPERRES_MAIN_JSON
    FRONTEND "artic"
    NAME "test"
    EMIT_JSON
    FILES test.art ${SUPPORT_FILES}
)

anydsl_runtime_wrap(TEST_CONV_SUPERRES_MAIN_INT
    FRONTEND "anyopt"
    ANYOPT_FLAGS --remove-interns
    EMIT_JSON
    NAME "test-compiled"
    FILES ${TEST_CONV_SUPERRES_MAIN_JSON}
)

anydsl_runtime_wrap(TEST_CONV_SUPERRES_OBJ
    FRONTEND "anyopt"
    PLUGINS plugin/load_plugin.cpp
    FILES ${TEST_CONV_SUPERRES_MAIN_INT}
)

add_executable(test_super_resolution ${TEST_CONV_SUPERRES_OBJ} ${PROJECT_SOURCE_DIR}/src/utils/read.cpp ${PROJECT_SOURCE_DIR}/src/utils/allocator.cpp)
target_link_libraries(test_super_resolution loader_runtime ${AnyDSL_runtime_LIBRARIES})

add_test(NAME super_resolution COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/test.py ${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})


# Generate test data on the fly
add_custom_target(CONV_SUPERRES_TEST_IDX
    COMMAND ${VENV}/bin/python ${CMAKE_CURRENT_SOURCE_DIR}/generate_data.py
    DEPENDS generate_data.py ${VENV}
)

add_test(build_super_resolution_testidx
  "${CMAKE_COMMAND}"
  --build "${CMAKE_BINARY_DIR}"
  --config "$<CONFIG>"
  --target CONV_SUPERRES_TEST_IDX
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/super-resolution-10.onnx DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(build_super_resolution_testidx PROPERTIES FIXTURES_SETUP super_resolution_fixture)
set_tests_properties(super_resolution PROPERTIES FIXTURES_REQUIRED super_resolution_fixture)

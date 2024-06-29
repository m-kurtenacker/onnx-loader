function(anydsl_runtime_fancy_wrap outfiles)
    cmake_parse_arguments(
        "PARGS"
        "EMIT_C;EMIT_JSON"
        "FRONTEND;INTERFACE;NAME"
        "ARTIC_FLAGS;ANYOPT_FLAGS;CLANG_FLAGS;HLS_FLAGS;PLUGINS;FILES" ${ARGN})
    if(NOT "${PARGS_UNPARSED_ARGUMENTS}" STREQUAL "")
        message(FATAL_ERROR "Unparsed arguments ${PARGS_UNPARSED_ARGUMENTS}")
    endif()

    set(_additional_platform_files)
    set(_plugin_flags)
    set(_opt_flags)

    if (NOT PARGS_EMIT_JSON)
        set(_opt_flags ${OPT_FLAGS})
    endif()

    foreach (_plugin_file ${PARGS_PLUGINS})
        get_filename_component(_plugin_basename ${_plugin_file} NAME_WE)
        set(_plugin_target ${_plugin_basename}_TARGET) #TODO: This fails if the same filename is used in different subfolders!

        #message(WARNING "plugin is ${_plugin_file}")
        #message(WARNING "target is ${_plugin_target}")

        #_plugin_file is a full path here.
        if (NOT TARGET ${_plugin_target})
            add_library(${_plugin_target} SHARED ${_plugin_file})

            set_target_properties(${_plugin_target} PROPERTIES PREFIX "" CXX_STANDARD 17)

            target_link_libraries(${_plugin_target} PUBLIC ${Thorin_LIBRARIES})
            target_include_directories(${_plugin_target} PUBLIC ${Thorin_INCLUDE_DIRS})
        endif()

        list(APPEND _plugin_flags --plugin $<TARGET_FILE:${_plugin_target}>)

        unset(_plugin_basename)
        unset(_plugin_target)
    endforeach()

    if(NOT PARGS_FRONTEND)
        set(_frontend "artic")
    else()
        string(TOLOWER ${PARGS_FRONTEND} _frontend)
    endif()

    if(${_frontend} STREQUAL "artic")
        # check for artic in toolchain
        if(NOT Artic_BIN)
            message(FATAL_ERROR "Could not find artic binary, please set Artic_DIR or Artic_BIN respectively")
        endif()
        set(_frontend_flags ${PARGS_ARTIC_FLAGS} ${_opt_flags} ${_plugin_flags})
        set(_frontend_bin ${Artic_BIN})
        list(APPEND _additional_platform_files "${AnyDSL_runtime_ROOT_DIR}/platforms/artic/intrinsics_math.impala")
    endif()

    if(${_frontend} STREQUAL "anyopt")
        # check for impala in toolchain
        if(NOT Anyopt_BIN)
            message(FATAL_ERROR "Could not find anyopt binary, please set Anyopt_DIR or Anyopt_BIN respectively")
        endif()
        set(_frontend_flags ${PARGS_ANYOPT_FLAGS} ${_opt_flags} ${_plugin_flags})
        set(_frontend_bin ${Anyopt_BIN})
    endif()

    # parse extra clang flags
    set(_clang_flags ${PARGS_CLANG_FLAGS} ${_opt_flags})

    list(FIND _frontend_flags "--log-level" FRONTEND_FLAGS_LOG_LEVEL_IDX)
    if(FRONTEND_FLAGS_LOG_LEVEL_IDX EQUAL -1)
        list(APPEND _frontend_flags --log-level $<IF:$<CONFIG:Release>,error,info>)
    endif()

    # check for clang in toolchain
    if(NOT Clang_BIN)
        message(FATAL_ERROR "Could not find clang binary, it has to be in the PATH")
    endif()

    # get last filename and absolute filenames
    set(_infiles)
    foreach(_it ${PARGS_FILES})
        get_filename_component(_infile ${_it} ABSOLUTE)
        set(_infiles ${_infiles} ${_infile})
        set(_lastfile ${_it})
    endforeach()

    if(NOT PARGS_NAME)
        get_filename_component(_basename ${_lastfile} NAME_WE)
    else()
        set(_basename ${PARGS_NAME})
    endif()

    if(NOT PARGS_HLS_FLAGS)
        set(HLS_COMMAND)
    else()
        string(REPLACE ";" "," HLS_FLAGS "${PARGS_HLS_FLAGS}")
        set(HLS_COMMAND COMMAND ${CMAKE_COMMAND} -D_basename=${_basename} -DHLS_FLAGS=${HLS_FLAGS} -P ${AnyDSL_runtime_DIR}/build_xilinx_hls.cmake)
    endif()

    set(_basepath ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${_basename})
    set(_llfile ${_basepath}.ll)
    set(_cfile ${_basepath}.c)
    set(_jsonfile ${_basepath}.thorin.json)
    set(_objfile ${_basepath}.o)

    if (_frontend STREQUAL "anyopt")
        set(_frontend_platform_files "")
    else()
        set(_frontend_platform_files
            ${AnyDSL_runtime_ROOT_DIR}/platforms/${_frontend}/intrinsics.impala
            ${AnyDSL_runtime_ROOT_DIR}/platforms/${_frontend}/intrinsics_rv.impala
            ${AnyDSL_runtime_ROOT_DIR}/platforms/${_frontend}/intrinsics_cpu.impala
            ${AnyDSL_runtime_ROOT_DIR}/platforms/${_frontend}/intrinsics_hls.impala
            ${AnyDSL_runtime_ROOT_DIR}/platforms/${_frontend}/intrinsics_cuda.impala
            ${AnyDSL_runtime_ROOT_DIR}/platforms/${_frontend}/intrinsics_nvvm.impala
            ${AnyDSL_runtime_ROOT_DIR}/platforms/${_frontend}/intrinsics_amdgpu.impala
            ${AnyDSL_runtime_ROOT_DIR}/platforms/${_frontend}/intrinsics_opencl.impala
            ${AnyDSL_runtime_ROOT_DIR}/platforms/${_frontend}/intrinsics_thorin.impala
            ${AnyDSL_runtime_ROOT_DIR}/platforms/${_frontend}/runtime.impala
            ${_additional_platform_files})
    endif()

    if(NOT MSVC)
        list(APPEND _clang_flags -fPIE)
    endif()

    if(PARGS_EMIT_C)
        # generate .c file
        add_custom_command(OUTPUT ${_cfile}
            COMMAND ${_frontend_bin} ${_frontend_platform_files} ${_infiles} ${_frontend_flags} --emit-c -o ${_basepath}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS ${_frontend_bin} ${_frontend_platform_files} ${_infiles} VERBATIM COMMAND_EXPAND_LISTS)
        # run clang on the .c file to get the .o
        add_custom_command(OUTPUT ${_objfile}
            COMMAND ${Clang_BIN} ${_clang_flags} -c -o ${_objfile} ${_cfile}
            DEPENDS ${_cfile} VERBATIM COMMAND_EXPAND_LISTS)
    elseif(PARGS_EMIT_JSON)
        # generate .json file
        add_custom_command(OUTPUT ${_jsonfile}
            COMMAND ${_frontend_bin} ${_frontend_platform_files} ${_infiles} ${_frontend_flags} --emit-json -o ${_basepath}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS ${_frontend_bin} ${_frontend_platform_files} ${_infiles} VERBATIM COMMAND_EXPAND_LISTS)
    else()
        # generate .ll file and patch it
        add_custom_command(OUTPUT ${_llfile}
            COMMAND ${_frontend_bin} ${_frontend_platform_files} ${_infiles} ${_frontend_flags} --emit-llvm -o ${_basepath}
            COMMAND ${Python3_EXECUTABLE} ${AnyDSL_runtime_ROOT_DIR}/post-patcher.py ${_basepath}
            COMMAND ${CMAKE_COMMAND} -D_basename=${_basename} -DLLVM_AS_BIN=${LLVM_AS_BIN} -P ${AnyDSL_runtime_ROOT_DIR}/cmake/check_nvvmir.cmake
            ${HLS_COMMAND}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS ${_frontend_bin} ${Python3_EXECUTABLE} ${AnyDSL_runtime_ROOT_DIR}/post-patcher.py ${_frontend_platform_files} ${_infiles} VERBATIM COMMAND_EXPAND_LISTS)
        # run clang on the .ll file to get the .o
        add_custom_command(OUTPUT ${_objfile}
            COMMAND ${Clang_BIN} ${_clang_flags} -c -o ${_objfile} ${_llfile}
            DEPENDS ${_llfile} VERBATIM COMMAND_EXPAND_LISTS)
    endif()

    # generate C interface on request
    if(NOT ${PARGS_INTERFACE} STREQUAL "")
        set(_hfile ${PARGS_INTERFACE}.h)
        add_custom_command(OUTPUT ${_hfile}
           COMMAND ${_frontend_bin} ${_frontend_platform_files} ${_infiles} --emit-c-interface -o ${PARGS_INTERFACE}
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
           DEPENDS ${_frontend_bin} ${_frontend_platform_files} ${_infiles} VERBATIM)
        set_source_files_properties(${_hfile} PROPERTIES GENERATED TRUE)
    endif()

    if(PARGS_EMIT_JSON)
        set_source_files_properties(${_jsonfile} PROPERTIES EXTERNAL_OBJECT true GENERATED true)
        set(${outfiles} ${${outfiles}} ${_jsonfile} PARENT_SCOPE)
    else()
        set_source_files_properties(${_objfile} PROPERTIES EXTERNAL_OBJECT true GENERATED true LANGUAGE CXX)
        set(${outfiles} ${${outfiles}} ${_objfile} ${_frontend_platform_files} PARENT_SCOPE)
    endif()
endfunction()

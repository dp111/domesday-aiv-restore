# Domesday model decoder (newdecode) bridge sink for decode-orc.
#
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<decode-orc install prefix>
#   cmake --build build
#   ORC_STAGE_PLUGIN_PATHS=build/lib/orc-stage-plugins orc-cli stages
#
# POSIX only (the decoder is run with fork/exec).

cmake_minimum_required(VERSION 3.20)
project(orc-plugin-domesday-model-decoder VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(decode-orc-plugin-sdk REQUIRED)

orc_add_stage_plugin(
    orc-plugin-domesday-model-decoder
    OUTPUT_NAME    orc-plugin_domesday-model-decoder
    PLUGIN_VERSION "${PROJECT_VERSION}"
    SOURCES        plugin.cpp domesday_model_decoder_stage.cpp ld_export.cpp
                   subprocess_runner.cpp
)

option(DOMESDAY_PLUGIN_TESTS "Build the unit tests (needs GTest)" ON)
if(DOMESDAY_PLUGIN_TESTS)
    find_package(GTest)
    if(GTest_FOUND)
        enable_testing()
        add_executable(domesday_plugin_tests
            tests/ld_export_test.cpp tests/stage_test.cpp
            ld_export.cpp domesday_model_decoder_stage.cpp subprocess_runner.cpp)
        target_include_directories(domesday_plugin_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(domesday_plugin_tests PRIVATE
            orc::plugin-sdk GTest::gtest_main)
        include(GoogleTest)
        gtest_discover_tests(domesday_plugin_tests)
    else()
        message(STATUS "GTest not found: unit tests skipped")
    endif()
endif()

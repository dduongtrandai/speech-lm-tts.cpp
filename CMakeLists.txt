cmake_minimum_required(VERSION 3.14)
project(speechlm-tts-project LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)

# DLL/Output Directory setup (same as build root)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# MSVC specific compile settings
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
endif()

# Configuration Options
set(SLM_LLAMA_DIR "../llama.cpp" CACHE PATH "Path to llama.cpp source directory")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime SDK root")

# Fallback environment check for ONNX Runtime
if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()

# Find llama.cpp dependency
if(NOT IS_ABSOLUTE "${SLM_LLAMA_DIR}")
    get_filename_component(SLM_LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${SLM_LLAMA_DIR}" ABSOLUTE)
endif()
if(NOT EXISTS "${SLM_LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp CMakeLists.txt not found at ${SLM_LLAMA_DIR}")
endif()

set(GGML_NATIVE OFF CACHE BOOL "Build portable CPU kernels instead of host-native kernels" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Do not build llama.cpp tests" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Do not build llama.cpp tools" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Do not build llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Do not build llama.cpp server" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "Do not build llama.cpp app" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "Do not build llama.cpp UI" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "Do not fetch llama.cpp prebuilt UI" FORCE)
add_subdirectory(${SLM_LLAMA_DIR} llama_cpp_build EXCLUDE_FROM_ALL)

# Find ONNX Runtime dependency
if(ONNXRUNTIME_ROOT)
    include_directories(SYSTEM "${ONNXRUNTIME_ROOT}/include")
    link_directories("${ONNXRUNTIME_ROOT}/lib")
    find_library(ONNXRUNTIME_LIB NAMES onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)
    message(STATUS "ONNX Runtime library found at: ${ONNXRUNTIME_LIB}")
else()
    find_library(ONNXRUNTIME_LIB NAMES onnxruntime REQUIRED)
endif()

# Core static library
add_library(speechlm-tts-core STATIC
    src/llm/llama_backend.cpp
    src/codecs/neucodec_onnx.cpp
    src/profiles/neutts_air.cpp
    src/profiles/vieneu.cpp
)

target_include_directories(speechlm-tts-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(speechlm-tts-core PUBLIC
    llama
    ${ONNXRUNTIME_LIB}
)

# Shared library for ABI consumers (speechlm-tts.dll)
add_library(speechlm-tts SHARED
    src/speechlm_tts.cpp
)
target_link_libraries(speechlm-tts PUBLIC speechlm-tts-core)
target_include_directories(speechlm-tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Export options for shared library on Windows
if(MSVC)
    target_compile_definitions(speechlm-tts PRIVATE SLM_BUILD_DLL)
endif()

# CLI Executable
add_executable(speechlm-tts-cli
    tools/speechlm-tts-cli.cpp
)
target_link_libraries(speechlm-tts-cli PRIVATE speechlm-tts)

# C ABI verification executable
add_executable(test-abi-c
    tests/abi-c.c
)
set_target_properties(test-abi-c PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
)
if(MSVC)
    target_compile_options(test-abi-c PRIVATE /W4 /WX)
else()
    target_compile_options(test-abi-c PRIVATE -Wall -Werror -pedantic)
endif()
target_include_directories(test-abi-c PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test-abi-c PRIVATE speechlm-tts)

if(BUILD_TESTING)
    add_executable(test-vieneu-phonemizer
        tests/vieneu-phonemizer.cpp
    )
    target_link_libraries(test-vieneu-phonemizer PRIVATE speechlm-tts-core)
    add_test(NAME vieneu-phonemizer COMMAND test-vieneu-phonemizer)
endif()

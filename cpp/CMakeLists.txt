cmake_minimum_required(VERSION 3.31...4.4)
project(CppPipeline
    VERSION 0.1.0
    DESCRIPTION "C++ pipeline for ONNX model comparison"
    LANGUAGES CXX
)

# Enforce precise floating-point behavior and prevent math reordering
if(MSVC)
    # Microsoft clang-cl
    add_compile_options(
        /fp:precise # keeps floating-point math exact: no reordering and no shortcuts.
        /EHsc # turns C++ exceptions on. ONNX Runtime expects them.
        /GR # turns RTTI on. RTTI is the runtime type information that dynamic_cast and typeid need. ONNX Runtime expects it.
        /clang:-ffp-contract=off #never fuses a multiply and an add into one multiply-add instruction. /clang: passes the flag through to clang.
    )
else()
    # GCC and clang++ (GNU-style drivers) need explicit flags for the same behavior.
    add_compile_options(
        -fno-associative-math # never regroups arithmetic. For example, it never turns (a + b) + c into a + (b + c).
        -frounding-math # respects the rounding mode. The compiler must not assume round-to-nearest.
        -ffp-contract=off # never fuses a multiply and an add into one multiply-add instruction.
    )
endif()

# ONNX Runtime
include(FetchContent)
FetchContent_Declare(onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-win-x64-1.31.0.zip
    URL_HASH SHA256=3958d8a44984160692c1b4ffd0de3cb48f8c39f9d34bb31e433db06e30b82195
)
FetchContent_MakeAvailable(onnxruntime)   # downloads only: the archive has no CMakeLists.txt
add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES 
    IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll"  # the DLL is in the lib subdirectory of the archive
    IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
)

# OpenCV: image decoding, color conversion, and resize. The official Windows
# package is a self-extracting 7z archive, so download it without extraction
# and unpack it once into the build tree.
FetchContent_Declare(opencv
    URL https://github.com/opencv/opencv/releases/download/5.0.0/opencv-5.0.0-windows.exe
    URL_HASH SHA256=9c6c1fcea58acdf06edba13148b2246e00c2658143fa51e61ecd370db8c39f63
    DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_MakeAvailable(opencv)   # downloads only: the package has no CMakeLists.txt
set(OPENCV_EXTRACT_DIR "${CMAKE_BINARY_DIR}/_deps/opencv-extracted")
set(OPENCV_CONFIG "${OPENCV_EXTRACT_DIR}/opencv/build/OpenCVConfig.cmake")
if(NOT EXISTS "${OPENCV_CONFIG}")
    execute_process(
        COMMAND "${opencv_SOURCE_DIR}/opencv-5.0.0-windows.exe"
                -y "-o${OPENCV_EXTRACT_DIR}"
        RESULT_VARIABLE opencv_extract_result
        OUTPUT_VARIABLE opencv_extract_output
        ERROR_VARIABLE opencv_extract_error
    )
    if(NOT opencv_extract_result EQUAL 0 OR NOT EXISTS "${OPENCV_CONFIG}")
        message(FATAL_ERROR
            "OpenCV extraction failed: ${opencv_extract_output}${opencv_extract_error}")
    endif()
endif()
find_package(OpenCV CONFIG REQUIRED
    PATHS "${OPENCV_EXTRACT_DIR}/opencv/build"
    NO_DEFAULT_PATH
)
# The package exports Release and Debug, while this project builds RelWithDebInfo.
set_target_properties(opencv_world PROPERTIES
    MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
)

# miniz: ZIP container for `outputs.npz`. The release archive has no
# CMakeLists.txt, so use the pinned source archive from the tagged revision.
FetchContent_Declare(miniz
    URL https://github.com/richgel999/miniz/archive/refs/tags/3.1.2.zip
    URL_HASH SHA256=e7a0334927a069dc5aeb69616d371542a4376e1569290d7102b88c6f3992d6cb
)
FetchContent_MakeAvailable(miniz)

# nlohmann/json: model settings and `meta.json`.
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
)
FetchContent_MakeAvailable(nlohmann_json)


add_executable(cpp_pipeline)
target_sources(cpp_pipeline
    PRIVATE 
        cpp_pipeline/main.cpp
        cpp_pipeline/cli.cpp
        cpp_pipeline/log.cpp
        cpp_pipeline/meta_json.cpp
        cpp_pipeline/results.cpp
        cpp_pipeline/npz.cpp
)

add_library(pipeline_core)
add_library(pipeline_core::pipeline_core 
    ALIAS 
        pipeline_core
)
target_sources(pipeline_core
    PRIVATE 
        pipeline_core/src/config.cpp
        pipeline_core/src/machine.cpp
        pipeline_core/src/memory.cpp
        pipeline_core/src/pipeline.cpp
        pipeline_core/src/postprocess.cpp
        pipeline_core/src/preprocess.cpp
        pipeline_core/src/run_support.cpp
        pipeline_core/src/session_contract.cpp
        pipeline_core/src/stub.cpp
    
    PUBLIC 
        FILE_SET HEADERS 
        BASE_DIRS pipeline_core/include 
        FILES 
            pipeline_core/include/pipeline_core/array.h
            pipeline_core/include/pipeline_core/core.h
            pipeline_core/include/pipeline_core/machine.h
            pipeline_core/include/pipeline_core/memory.h
            pipeline_core/include/pipeline_core/pipeline.h
            pipeline_core/include/pipeline_core/results.h
            pipeline_core/include/pipeline_core/run.h
            pipeline_core/include/pipeline_core/stub.h
)
target_include_directories(pipeline_core
    PUBLIC
        pipeline_core/include
)

target_link_libraries(cpp_pipeline
    PRIVATE 
        pipeline_core
        miniz
        nlohmann_json::nlohmann_json
)

target_link_libraries(pipeline_core
    PRIVATE 
        onnxruntime::onnxruntime
        opencv_world
        nlohmann_json::nlohmann_json
        advapi32
        psapi
)

add_custom_command(TARGET cpp_pipeline POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:onnxruntime::onnxruntime>"
        "$<TARGET_FILE_DIR:cpp_pipeline>/onnxruntime.dll"
)

add_custom_command(TARGET cpp_pipeline POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:opencv_world>"
        "$<TARGET_FILE_DIR:cpp_pipeline>/$<TARGET_FILE_NAME:opencv_world>"
)

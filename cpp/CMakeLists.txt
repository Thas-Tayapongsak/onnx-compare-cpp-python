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


add_executable(cpp_pipeline)
target_sources(cpp_pipeline
    PRIVATE 
        cpp_pipeline/main.cpp
)

add_library(pipeline_core)
add_library(pipeline_core::pipeline_core 
    ALIAS 
        pipeline_core
)
target_sources(pipeline_core
    PRIVATE 
        pipeline_core/src/core.cpp 
    
    PUBLIC 
        FILE_SET HEADERS 
        BASE_DIRS pipeline_core/include 
        FILES 
            pipeline_core/include/pipeline_core/core.h
)

target_link_libraries(cpp_pipeline
    PRIVATE 
        pipeline_core
)

target_link_libraries(pipeline_core
    PRIVATE 
        onnxruntime::onnxruntime
)
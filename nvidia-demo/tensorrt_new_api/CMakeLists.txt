cmake_minimum_required(VERSION 3.15)

project(tensorrt_new_api LANGUAGES CXX C CUDA)

add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_DEBUG_POSTFIX "d") 
if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CompilerFlags
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        )
        foreach(CompilerFlag ${CompilerFlags})
            string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
        endforeach()
endif(MSVC)

file(GLOB SOURCES ${CMAKE_CURRENT_LIST_DIR}/srcs/*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_LIST_DIR}/includes/*.h)

if(DEFINED_TENSORRT)
    message(STATUS "the TensorRT config has been loaded")
else()
    set(DEFINED_TENSORRT 1)
    set(TENSORRT_ROOT "D:\\3rdparty_for_1.0\\TensorRT-8.6.1.6")
    message(STATUS "TENSORRT_ROOT ${TENSORRT_ROOT}")

    set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
    if (WIN32)
        set(TENSORRT_LIBRARY_DIR ${TENSORRT_ROOT}/lib/windows)
    else()
        set(TENSORRT_LIBRARY_DIR ${TENSORRT_ROOT}/lib/linux)
    endif()

    set(LIBRARIES_TO_FIND nvinfer)
    foreach(LIB ${LIBRARIES_TO_FIND})
        find_library(${LIB}_PATH ${LIB} HINTS ${TENSORRT_LIBRARY_DIR})
        if(${LIB}_PATH)
            message("Found ${LIB} at ${${LIB}_PATH}")
            list(APPEND TENSORRT_LIBS ${${LIB}_PATH})
        else()
            message("Could not find ${LIB}")
        endif()
    endforeach()
    message(STATUS "TENSORRT LIBS: ${TENSORRT_LIBS}")
endif()

if(DEFINED_CUDA)
    message(STATUS "the CUDA config has been loaded")
else()
    set(DEFINED_CUDA 1)
    if(WIN32)
        set(CUDA_ROOT "D:\\3rdparty_for_1.0\\cuda-11.8\\windows")
    else()
        set(CUDA_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../timesAI_platform/times-ai/3rdparty/cuda-11.8/linux)
    endif()
    message(STATUS "CUDA_ROOT ${CUDA_ROOT}")

    set(CUDA_INCLUDE_DIR ${CUDA_ROOT}/include)
    set(CUDA_LIBRARY_DIR ${CUDA_ROOT}/lib/x64)

    set(LIBRARIES_TO_FIND cuda cublas cublasLt cudart cudadevrt)
    foreach(LIB ${LIBRARIES_TO_FIND})
        find_library(${LIB}_PATH ${LIB} HINTS ${CUDA_LIBRARY_DIR})
        if(${LIB}_PATH)
            message("Found ${LIB} at ${${LIB}_PATH}")
            list(APPEND CUDA_LIBS ${${LIB}_PATH})
        else()
            message("Could not find ${LIB}")
        endif()
    endforeach()
    message(STATUS "CUDA LIBS: ${CUDA_LIBS}")
endif()
unset(LIBRARIES_TO_FIND)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -g -D_MWAITXINTRIN_H_INCLUDED -DNOMINMAX")
add_executable(tensorrt_new_api main.cpp ${SOURCES} ${HEADERS})
target_link_directories(tensorrt_new_api PUBLIC "includes")
target_compile_options(tensorrt_new_api PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>)
target_link_libraries(tensorrt_new_api ${CUDA_LIBS} ${TENSORRT_LIBS})
target_include_directories(tensorrt_new_api PUBLIC "includes" ${CUDA_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})

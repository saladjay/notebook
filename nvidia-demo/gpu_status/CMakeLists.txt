cmake_minimum_required(VERSION 3.20)

project(gpu_status LANGUAGES CXX C CUDA)

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
file(GLOB CU_FILES ${CMAKE_CURRENT_LIST_DIR}/srcs/*.cu)
file(GLOB HEADERS ${CMAKE_CURRENT_LIST_DIR}/includes/*.h)
message(STATUS ${SOURCES})
message(STATUS ${HEADERS})
set(CMAKE_CUDA_COMPILER C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/bin/nvcc.exe)
link_directories(C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/lib/x64)
include_directories(C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/include)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -g -D_MWAITXINTRIN_H_INCLUDED -DNOMINMAX")
add_executable(gpu_status ${CU_FILES} ${HEADERS} ${SOURCES})
target_compile_options(gpu_status PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>)
set_target_properties(gpu_status PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(gpu_status cudart)

# add_executable(two_stream ${SOURCES} ${HEADERS})
# target_link_libraries(two_stream two_stream_lib)
# target_link_libraries(two_stream cudart)
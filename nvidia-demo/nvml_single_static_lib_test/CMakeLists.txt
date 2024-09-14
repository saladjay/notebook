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
set(SOURCES main.cpp)




# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -g -D_MWAITXINTRIN_H_INCLUDED -DNOMINMAX")
add_executable(nvml_single_static_lib_test ${SOURCES})
target_include_directories(nvml_single_static_lib_test PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(nvml_single_static_lib_test "D:\\github\\notebook\\nvidia-demo\\nvml_single_static_lib_test\\nvml.lib")

# add_executable(two_stream ${SOURCES} ${HEADERS})
# target_link_libraries(two_stream two_stream_lib)
# target_link_libraries(two_stream cudart)
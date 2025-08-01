cmake_minimum_required(VERSION 3.10)
project(wrap_cuda_malloc)
# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加拦截库
add_subdirectory(intercept)

# 添加主程序
add_executable(wrap_cuda_malloc main.cpp)

# 链接拦截库（可选，这里只是为了确保顺序）
target_link_libraries(${PROJECT_NAME}  intercept_lib)

set(CUDA_ROOT "/usr/local/cuda/")
set(CUDA_INCLUDE_DIR ${CUDA_ROOT}/include)
set(CUDA_LIBRARY_DIR ${CUDA_ROOT}/lib64)
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
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBS})

# 设置CUDA编译选项（可选）
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86"  # 根据你的GPU架构调整
)
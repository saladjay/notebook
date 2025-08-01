# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
enable_language(CUDA)
# 找到CUDA工具包
find_package(CUDA REQUIRED)

# 创建共享库
add_library(intercept_lib SHARED intercept.cpp)

# 链接 dl 库（用于 dlsym）
target_link_libraries(intercept_lib dl)

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
target_include_directories(intercept_lib PUBLIC ${CUDA_INCLUDE_DIR})
target_link_libraries(intercept_lib ${CUDA_LIBS})

# 设置CUDA编译选项（可选）
set_target_properties(intercept_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86"  # 根据你的GPU架构调整
)
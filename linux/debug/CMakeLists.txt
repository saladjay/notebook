cmake_minimum_required(VERSION 3.10)
project(LD_PRELOAD_Example)

# 添加拦截库
add_subdirectory(intercept)

# 添加主程序
add_executable(main_program main.c)

# 链接拦截库（可选，这里只是为了确保顺序）
# target_link_libraries(main_program intercept_lib)
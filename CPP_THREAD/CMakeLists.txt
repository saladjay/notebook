cmake_minimum_required(VERSION 3.10)
project(CPP_THREAD)

set(CMAKE_CXX_STANDARD 17)

add_executable(cpp_thread main.cpp) 
target_link_libraries(cpp_thread pthread)
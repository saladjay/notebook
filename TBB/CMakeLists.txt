cmake_minimum_required(VERSION 3.10)
project(TBB)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(oneTBB)

add_executable(test main.cpp)
# target_link_libraries(test oneTBB::tbb)
# target_include_directories(test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/oneTBB/include)
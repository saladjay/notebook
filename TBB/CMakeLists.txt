cmake_minimum_required(VERSION 3.10)
project(TBB)

set(CMAKE_CXX_STANDARD 17)


# 添加模组的地址
set(TBB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/TBB/lib/cmake/TBB")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/TBB/lib/cmake/TBB")
include(TBBConfig)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/DPL/lib/cmake/oneDPL")
include(oneDPLConfig)

get_target_property(oneDPL_INCLUDE_HEADER oneDPL INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "DPL_INCLUDE_HEADER:${oneDPL_INCLUDE_HEADER}")


add_executable(test_abc main.cpp)
target_include_directories(test_abc PRIVATE TBB::tbb oneDPL)
target_link_libraries(test_abc PRIVATE TBB::tbb oneDPL)


add_subdirectory(new_example)
# # set(dependencies_to_link tbb12 tbbmalloc tbbmalloc_proxy)
# # set(dependencies_to_link_list)
# # foreach(dep ${dependencies_to_link})
# #     # 在文件夹里找到依赖的库文件
# #     find_library(${dep}_LIBRARY NAMES ${dep} PATHS ${CMAKE_CURRENT_SOURCE_DIR}/TBB/lib)
# #     # 将依赖的库文件一个集合里面
# #     list(APPEND dependencies_to_link_list ${${dep}_LIBRARY})
# # endforeach()

# target_link_libraries(test_abc INTERFACE TBB::tbb)
# # target_link_libraries(test_abc ${dependencies_to_link_list})
# target_include_directories(test_abc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/TBB/include)
# target_include_directories(test_abc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/oneDPL/include/oneapi/dpl)
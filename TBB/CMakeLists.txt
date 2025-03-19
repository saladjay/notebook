cmake_minimum_required(VERSION 3.10)
project(TBB)

set(CMAKE_CXX_STANDARD 17)

# # 添加oneTBB和oneDPL文件夹
# add_subdirectory(oneTBB)
# # 将oneTBB安装到指定目录
# set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/build/msvc_19.16_cxx17_64_md_release)
# install(DIRECTORY oneTBB/include/oneapi/tbb/ DESTINATION include/oneapi/tbb)
# install(DIRECTORY oneTBB/lib/ DESTINATION lib)
# install(DIRECTORY oneTBB/bin/ DESTINATION bin)
# install(DIRECTORY oneTBB/cmake/ DESTINATION cmake)
# install(DIRECTORY oneTBB/share/ DESTINATION share)
# install(DIRECTORY oneTBB/doc/ DESTINATION doc)
# # 设置FindTBB的路径
# set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/oneTBB/cmake)
# find_package(TBB REQUIRED)
# include_directories(${TBB_INCLUDE_DIRS})
# link_directories(${TBB_LIBRARY_DIRS})
# add_subdirectory(oneDPL)
add_subdirectory(workflow)
# add_executable(test main.cpp)

# set(dependencies_to_link tbb12 tbbmalloc tbbmalloc_proxy)
# set(dependencies_to_link_list)
# foreach(dep ${dependencies_to_link})
#     # 在文件夹里找到依赖的库文件
#     find_library(${dep}_LIBRARY NAMES ${dep} PATHS ${CMAKE_CURRENT_SOURCE_DIR}/build/msvc_19.16_cxx17_64_md_release)
#     # 将依赖的库文件一个集合里面
#     list(APPEND dependencies_to_link_list ${${dep}_LIBRARY})
# endforeach()


# target_link_libraries(test ${dependencies_to_link_list})
# target_include_directories(test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/oneTBB/include)
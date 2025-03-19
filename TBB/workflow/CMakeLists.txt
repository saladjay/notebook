project(workflow)

set(CMAKE_CXX_STANDARD 17)

add_executable(workflow main.cpp)
# target_link_libraries(workflow oneTBB::tbb)
# target_link_directories(workflow PUBLIC ${TBB_LIBRARY_DIRS})
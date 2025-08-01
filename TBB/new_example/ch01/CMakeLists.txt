
project(ch01)

add_executable(fig_1_04 fig_1_04.cpp)
target_include_directories(fig_1_04 PRIVATE TBB::tbb oneDPL)
target_link_libraries(fig_1_04 PRIVATE TBB::tbb oneDPL)

add_executable(fig_1_05 fig_1_05.cpp)
target_include_directories(fig_1_05 PRIVATE TBB::tbb oneDPL)
target_link_libraries(fig_1_05 PRIVATE TBB::tbb oneDPL)

add_executable(fig_1_07 fig_1_07.cpp)
target_include_directories(fig_1_07 PRIVATE TBB::tbb oneDPL)
target_link_libraries(fig_1_07 PRIVATE TBB::tbb oneDPL)

add_executable(fig_1_08 fig_1_08.cpp)
target_include_directories(fig_1_08 PRIVATE TBB::tbb oneDPL)
target_link_libraries(fig_1_08 PRIVATE TBB::tbb oneDPL)

add_executable(fig_1_10 fig_1_10.cpp)
target_include_directories(fig_1_10 PRIVATE TBB::tbb oneDPL)
target_link_libraries(fig_1_10 PRIVATE TBB::tbb oneDPL)

add_executable(fig_1_11 fig_1_11.cpp)
target_include_directories(fig_1_11 PRIVATE TBB::tbb oneDPL)
target_link_libraries(fig_1_11 PRIVATE TBB::tbb oneDPL)

add_executable(fig_1_12 fig_1_12.cpp)
target_include_directories(fig_1_12 PRIVATE TBB::tbb oneDPL)
target_link_libraries(fig_1_12 PRIVATE TBB::tbb oneDPL)
cmake_minimum_required (VERSION 3.8)

project ("IsAPISetImplented")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_GENERATOR "Visual Studio 15 2017")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")


include_directories(D:\3rdparty\lua-5.4.2\windows\include)
link_directories(D:\3rdparty\lua-5.4.2\windows\bin)
add_executable(main main.cpp)
target_link_libraries(main lua54)
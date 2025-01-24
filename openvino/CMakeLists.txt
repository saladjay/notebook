project(openvino_study)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")

# 设置OpenVINO cmake路径， 加载Config.cmake文件
set(OpenVINO_DIR "E:/3rd_party/openvino_compile/OpenVINO/runtime/cmake")
find_package(OpenVINO REQUIRED)
# 验证是否加载成功
if(NOT OpenVINO_FOUND)
    message(FATAL_ERROR "OpenVINO not found")
else()
    message(STATUS "OpenVINO found: ${OpenVINO_DIR}")
    message(STATUS "OpenVINO version: ${OpenVINO_VERSION}")
    # 5. 打印所有 OpenVINO 相关变量
    get_cmake_property(_variableNames VARIABLES)
    foreach(_variableName ${_variableNames})
        if(_variableName MATCHES "^OpenVINO_.*")
            message(STATUS "${_variableName}=${${_variableName}}")
        endif()
    endforeach()

    # 6. 打印目标属性
    if(TARGET openvino::runtime)
        get_target_property(_props openvino::runtime INTERFACE_INCLUDE_DIRECTORIES)
        message(STATUS "OpenVINO Runtime include dirs: ${_props}")

        get_target_property(_location openvino::runtime LOCATION)
        message(STATUS "OpenVINO Runtime location: ${_location}")
    endif()


endif()

add_executable(openvino_study main.cpp block.cpp block.h) 
target_link_libraries(openvino_study PRIVATE openvino::runtime)
target_link_directories(openvino_study PRIVATE openvino::runtime)
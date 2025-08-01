##===----------------------------------------------------------------------===##
#
# Copyright (C) Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# This file incorporates work covered by the following copyright and permission
# notice:
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
#
##===----------------------------------------------------------------------===##

if (CMAKE_HOST_WIN32 AND CMAKE_VERSION VERSION_LESS 3.20)
    message(STATUS "oneDPL: requires CMake 3.20 or later on Windows")
    set(oneDPL_FOUND FALSE)
    return()
elseif (NOT CMAKE_HOST_WIN32 AND CMAKE_VERSION VERSION_LESS 3.11)
    message(STATUS "oneDPL: requires CMake 3.11 or later on Linux")
    set(oneDPL_FOUND FALSE)
    return()
endif()

if (CMAKE_HOST_WIN32)
    # Requires version 3.20 for baseline support of icx, icx-cl
    cmake_minimum_required(VERSION 3.20)
else()
    # Requires version 3.11 for use of icpc with c++17 requirement
    cmake_minimum_required(VERSION 3.11)
endif()

# Installation path: <onedpl_root>/lib/cmake/oneDPL/
get_filename_component(_onedpl_root "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
get_filename_component(_onedpl_root "${_onedpl_root}/../../../" ABSOLUTE)

get_filename_component(_onedpl_headers "${_onedpl_root}/include" ABSOLUTE)

if (EXISTS "${_onedpl_headers}")
    if (NOT TARGET oneDPL)
        if (CMAKE_CXX_COMPILER MATCHES ".*(dpcpp-cl|icx-cl|icx)(.exe)?$")
            set(INTEL_LLVM_COMPILER_MSVC_LIKE TRUE)
        elseif(CMAKE_CXX_COMPILER MATCHES ".*(dpcpp|icpx)(.exe)?$")
            set(INTEL_LLVM_COMPILER_GNU_LIKE TRUE)
        endif()

        if ((NOT oneDPLWindowsIntelLLVM_FOUND) AND CMAKE_HOST_WIN32 AND (INTEL_LLVM_COMPILER_MSVC_LIKE OR INTEL_LLVM_COMPILER_GNU_LIKE))
            message(WARNING "On Windows, ${CMAKE_CXX_COMPILER} requires some workarounds to function properly with CMake. We recommend using 'find_package(oneDPLWindowsIntelLLVM)' before your 'project()' call to apply these workarounds.")
        endif()

        include(CheckCXXCompilerFlag)
        include(CheckIncludeFileCXX)

        add_library(oneDPL INTERFACE IMPORTED)
        set_target_properties(oneDPL PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_onedpl_headers}")

        target_compile_features(oneDPL INTERFACE cxx_std_17)

        if (ONEDPL_PAR_BACKEND AND NOT ONEDPL_PAR_BACKEND MATCHES "^(tbb|openmp|serial)$")
            message(STATUS "oneDPL: ONEDPL_PAR_BACKEND=${ONEDPL_PAR_BACKEND} is requested, but not supported, available backends: tbb, openmp, serial")
            set(oneDPL_FOUND FALSE)
            return()
        endif()

        if (NOT ONEDPL_PAR_BACKEND OR ONEDPL_PAR_BACKEND STREQUAL "tbb")  # Handle oneTBB backend
            find_package(TBB 2021 QUIET COMPONENTS tbb)
            if (NOT TBB_FOUND AND ONEDPL_PAR_BACKEND STREQUAL "tbb")  # If oneTBB backend is requested explicitly, but not found.
                message(STATUS "oneDPL: ONEDPL_PAR_BACKEND=${ONEDPL_PAR_BACKEND} requested, but not found")
                set(oneDPL_FOUND FALSE)
                return()
            elseif (TBB_FOUND)
                set(ONEDPL_PAR_BACKEND tbb)
                message(STATUS "oneDPL: ONEDPL_PAR_BACKEND=${ONEDPL_PAR_BACKEND}, disable OpenMP backend")
                set_target_properties(oneDPL PROPERTIES INTERFACE_LINK_LIBRARIES TBB::tbb)
                set_property(TARGET oneDPL APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ONEDPL_USE_TBB_BACKEND=1 ONEDPL_USE_OPENMP_BACKEND=0)
            endif()
        endif()

        if (NOT ONEDPL_PAR_BACKEND OR ONEDPL_PAR_BACKEND STREQUAL "openmp")  # Handle OpenMP backend
            set(_openmp_minimum_version 4.5)
            find_package(OpenMP)
            if (OpenMP_CXX_FOUND)
                if (OpenMP_CXX_VERSION VERSION_LESS _openmp_minimum_version)
                    message(STATUS "oneDPL: OpenMP version (${OpenMP_CXX_VERSION}) is less than minimum supported version (${_openmp_minimum_version}), disable OpenMP" )
                else()
                    set(_openmp_found_valid TRUE)
                endif()
            endif()

            if (NOT _openmp_found_valid AND ONEDPL_PAR_BACKEND STREQUAL "openmp")  # If OpenMP backend is requested explicitly, but not supported.
                message(STATUS "oneDPL: ONEDPL_PAR_BACKEND=${ONEDPL_PAR_BACKEND} requested, but not supported")
                set(oneDPL_FOUND FALSE)
                return()
            elseif (_openmp_found_valid)
                set(ONEDPL_PAR_BACKEND openmp)
                message(STATUS "oneDPL: ONEDPL_PAR_BACKEND=${ONEDPL_PAR_BACKEND}, disable oneTBB backend")
                # Due to minor correctness issues with -fiopenmp / -Qiopenmp, we are using -fopenmp / -Qopenmp until they are corrected.
                # Once correctness issues are resolved, we will limit this workaround to affected versions of specific compilers.
                if (OpenMP_CXX_FLAGS MATCHES ".*-fiopenmp.*")
                    set(_openmp_flag -fopenmp)
                elseif (OpenMP_CXX_FLAGS MATCHES ".*[-/]Qiopenmp.*")
                    set(_openmp_flag /Qopenmp)
                endif()
                if (_openmp_flag)
                    message(STATUS "oneDPL: Using ${_openmp_flag} for openMP")
                    set_target_properties(oneDPL PROPERTIES INTERFACE_COMPILE_OPTIONS ${_openmp_flag})
                    set_target_properties(oneDPL PROPERTIES INTERFACE_LINK_LIBRARIES ${_openmp_flag})
                else()
                    set_target_properties(oneDPL PROPERTIES INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_CXX)
                endif()
                set_property(TARGET oneDPL APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ONEDPL_USE_TBB_BACKEND=0 ONEDPL_USE_OPENMP_BACKEND=1)
            endif()
        endif()

        if (NOT ONEDPL_PAR_BACKEND OR ONEDPL_PAR_BACKEND STREQUAL "serial")
            set(ONEDPL_PAR_BACKEND serial)
            message(STATUS "oneDPL: ONEDPL_PAR_BACKEND=${ONEDPL_PAR_BACKEND}, disable oneTBB and OpenMP backends")
            set_property(TARGET oneDPL APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ONEDPL_USE_TBB_BACKEND=0 ONEDPL_USE_OPENMP_BACKEND=0)
        endif()

        # Check SYCL support by the compiler
        set(FSYCL_OPTION "-fsycl")
        check_cxx_compiler_flag(${FSYCL_OPTION} _fsycl_option)
        if (_fsycl_option)
            set(FSYCL_OPTION_IF_SUPPORTED ${FSYCL_OPTION})
        endif()

        CHECK_INCLUDE_FILE_CXX("sycl/sycl.hpp" SYCL_HEADER ${FSYCL_OPTION_IF_SUPPORTED})
        if (NOT SYCL_HEADER)
            CHECK_INCLUDE_FILE_CXX("CL/sycl.hpp" SYCL_HEADER_OLD ${FSYCL_OPTION_IF_SUPPORTED})
        endif()
        if (SYCL_HEADER OR SYCL_HEADER_OLD)
            set(SYCL_SUPPORT TRUE)
        endif()

        if (SYCL_SUPPORT)
            # Enable SYCL* with compilers/compiler drivers not passing -fsycl by default
            if (_fsycl_option AND NOT CMAKE_CXX_COMPILER MATCHES ".*(dpcpp-cl|dpcpp)(.exe)?$")
                message(STATUS "Adding ${FSYCL_OPTION} compiler option")
                target_compile_options(oneDPL INTERFACE ${FSYCL_OPTION})
                target_link_libraries(oneDPL INTERFACE ${FSYCL_OPTION})
            endif()
        else()
            message(STATUS "oneDPL: SYCL is not supported. Set ONEDPL_USE_DPCPP_BACKEND=0")
            set_property(TARGET oneDPL APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ONEDPL_USE_DPCPP_BACKEND=0)
        endif()
    endif()
else()
    message(STATUS "oneDPL: headers do not exist ${_onedpl_headers}")
    set(oneDPL_FOUND FALSE)
endif()

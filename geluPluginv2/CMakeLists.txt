#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(gelu LANGUAGES CXX CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")


# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
find_package(CUDA REQUIRED)
find_package(CUDA REQUIRED)

#set_ifndef(TRT_LIB /usr/lib/x86_64-linux-gnu)
#set_ifndef(TRT_INCLUDE /usr/include/x86_64-linux-gnu)
#set_ifndef(CUDA_ROOT /usr/local/cuda)

#set_ifndef(TRT_LIB /home/codesteller/apsoft/mlstack/TensorRT-6.0.1.8/lib)
#set_ifndef(TRT_INCLUDE /home/codesteller/apsoft/mlstack/TensorRT-6.0.1.8/include)

set_ifndef(TRT_LIB /home/codesteller/apsoft/mlstack/tensorrt/TensorRT-7.0.0.11/lib)
set_ifndef(TRT_INCLUDE /home/codesteller/apsoft/mlstack/tensorrt/TensorRT-7.0.0.11/include)

set_ifndef(CUDA_ROOT /usr/local/cuda)

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

# cuBLAS
find_library(_CUBLAS_LIB cublas HINTS ${CUDA_ROOT} PATH_SUFFIXES lib lib64)
set_ifndef(CUBLAS_LIB ${_CUBLAS_LIB})

# CUDA include dir
find_path(_CUDA_INC_DIR cuda_runtime_api.h HINTS ${CUDA_ROOT} PATH_SUFFIXES include)
set_ifndef(CUDA_INC_DIR ${_CUDA_INC_DIR})

# -------- BUILDING --------
# include_directories(${TRT_INCLUDE} ${CUDA_INC_DIR})
include_directories(${CUDA_ROOT}/include  ${TRT_INCLUDE})
include_directories(${CUDA_ROOT}/include  ${TRT_INCLUDE} ${CMAKE_SOURCE_DIR}/include/ ${CMAKE_SOURCE_DIR}/) 

add_library(GeluPlugin MODULE
    ${CMAKE_SOURCE_DIR}/geluPlugin.cu
     ${CMAKE_SOURCE_DIR}/geluKernel.h
    ${CMAKE_SOURCE_DIR}/geluPlugin.cpp
    ${CMAKE_SOURCE_DIR}/geluPlugin.h
    
)

# Use C++11
target_compile_features(GeluPlugin PUBLIC cxx_std_11)

# Link TensorRT's nvinfer lib
target_link_libraries(GeluPlugin PRIVATE ${NVINFER_LIB} ${CUBLAS_LIB})

set_target_properties(GeluPlugin PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

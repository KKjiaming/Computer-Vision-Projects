cmake_minimum_required(VERSION 3.13)
project(HandsUpVotingDetectProject)

set(CMAKE_CXX_STANDARD 17)
if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")
if(${CMAKE_BUILD_TYPE} STREQUAL Release)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -fuse-ld=gold -Wall  -O3 ")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fuse-ld=gold -Wall  -O3 ")
endif()

set(InferModule_DIR ${PROJECT_SOURCE_DIR}/thirdParty/InferModule)
set(smdnnmodel_DIR ${PROJECT_SOURCE_DIR}/thirdParty/smdnnmodel)
set(ONNXRuntime_DIR ${PROJECT_SOURCE_DIR}/thirdParty/ONNXRuntime)
set(TensorRT_DIR ${PROJECT_SOURCE_DIR}/thirdParty/TensorRT)
set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/thirdParty/OpenCV/cmake)
set(HandsUpDetectModule_DIR ${PROJECT_SOURCE_DIR}/HandsUpDetectModule)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(InferModule REQUIRED)
find_package(smdnnmodel REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(ONNXRuntime REQUIRED)
find_package(TensorRT REQUIRED)

# SET OUTPUT PATH
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(HandsUpDetectModule)
add_subdirectory(tutorials)

include(${PROJECT_SOURCE_DIR}/HandsUpDetectModule/cuda.cmake NO_POLICY_SCOPE)

file(GLOB_RECURSE SRCS
        "backend_ops/*.cu"
        "backend_ops/*.cpp"
        "src/*.h"
        "src/*.c"
        "src/*.cpp"
        "include/*.h"
        "internal/*.cpp"
        "internal/*.h")

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(module HandsUpDetectModule)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/${module}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/${module}/lib)

add_library(${module} SHARED ${SRCS})

set(backend_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/backend_ops/common
        ${CMAKE_CURRENT_SOURCE_DIR}/backend_ops/tensorrt/common)

include_directories(${backend_INCLUDE_DIRS})

target_include_directories(${module}
PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/${module}/include>
PRIVATE
        ${InferModule_INCLUDE_DIRS}
        ${smdnnmodel_INCLUDE_DIRS}
        ${OpenMP_CXX_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${ONNXRuntime_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
)

target_link_libraries( ${module}
PUBLIC
        ${OpenCV_LIBRARIES}
PRIVATE
        ${InferModule_LIBRARIES}
        ${smdnnmodel_LIBRARIES}
        ${OpenMP_CXX_LIBRARIES}
        ${ONNXRuntime_LIBRARIES}
        ${TensorRT_LIBRARIES}
        ${CUDA_LIBRARIES}
        -lstdc++fs
)

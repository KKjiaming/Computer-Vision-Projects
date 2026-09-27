#find_path(HandsUpDetectModule_INCLUDE_DIRS
#        NAMES HandsUpDetectModule.h
#        PATHS ${HandsUpDetectModule_DIR}/include)
#
#find_library(HandsUpDetectModule_LIBRARY
#        NAMES HandsUpDetectModule
#        PATHS ${HandsUpDetectModule_DIR}/lib)

include_directories(${HandsUpDetectModule_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ONNXRuntime_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})

set(thirdPty_LIBRARIES
        ${InferModule_LIBRARIES}
        ${smdnnmodel_LIBRARIES}
        ${ONNXRuntime_LIBRARIES}
        ${TensorRT_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

set(DEMO_LIST
        demo
)

foreach(demo_name ${DEMO_LIST})
    add_executable(${demo_name} ${demo_name}.cc utility.cc utility.h)
    target_link_libraries(${demo_name} HandsUpDetectModule ${thirdPty_LIBRARIES} -lstdc++fs)
    add_dependencies(${demo_name} HandsUpDetectModule)
endforeach()


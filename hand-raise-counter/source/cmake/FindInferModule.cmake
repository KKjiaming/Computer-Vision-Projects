find_path(InferModule_INCLUDE_DIR
    NAMES InferModule/Tensor.h
    PATHS ${InferModule_DIR}/include)

find_library(InferModule_LIBRARY
    NAMES InferModule
    PATHS ${InferModule_DIR}/lib)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(InferModule DEFAULT_MSG
    InferModule_INCLUDE_DIR 
    InferModule_LIBRARY)

if(InferModule_FOUND)
    set(InferModule_INCLUDE_DIRS ${InferModule_INCLUDE_DIR})
    set(InferModule_LIBRARIES ${InferModule_LIBRARY})
    message(STATUS "Found InferModule: \n include: ${InferModule_INCLUDE_DIRS}\n library: ${InferModule_LIBRARIES}")
endif()
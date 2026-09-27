find_path(smdnnmodel_INCLUDE_DIR 
    NAMES smdnnmodel/dnnmodel.h
    PATHS ${smdnnmodel_DIR}/include)

find_library(smdnnmodel_LIBRARY
    NAMES smdnnmodel
    PATHS ${smdnnmodel_DIR}/lib)

find_library(sodium_LIBRARY
    NAMES sodium
    PATHS ${smdnnmodel_DIR}/lib)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(smdnnmodel DEFAULT_MSG
    smdnnmodel_INCLUDE_DIR 
    sodium_LIBRARY
    smdnnmodel_LIBRARY)

if(smdnnmodel_FOUND)
    set(smdnnmodel_INCLUDE_DIRS ${smdnnmodel_INCLUDE_DIR})
    set(smdnnmodel_LIBRARIES ${smdnnmodel_LIBRARY} ${sodium_LIBRARY})
    message(STATUS "Found smdnnmodel: \n include: ${smdnnmodel_INCLUDE_DIRS}\n library: ${smdnnmodel_LIBRARIES}")
endif()
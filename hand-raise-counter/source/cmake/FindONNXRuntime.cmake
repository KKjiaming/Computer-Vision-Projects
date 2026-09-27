find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS ${ONNXRuntime_DIR}/include)

find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS ${ONNXRuntime_DIR}/lib)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime DEFAULT_MSG ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)

if(ONNXRuntime_FOUND)
    set(ONNXRuntime_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
    set(ONNXRuntime_LIBRARIES ${ONNXRUNTIME_LIBRARY})
    if(WIN32)
        set(ONNXRuntime_IMPORTED_LOCATION ${ONNXRuntime_DIR}/lib/onnxruntime.dll)

        add_library(ONNXRuntime SHARED IMPORTED)
        set_target_properties(ONNXRuntime
        PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIRS}"
            IMPORTED_LOCATION ${ONNXRuntime_IMPORTED_LOCATION}
            IMPORTED_IMPLIB ${ONNXRuntime_LIBRARIES}
        )
    else()
        set(ONNXRuntime_IMPORTED_LOCATION ${ONNXRUNTIME_LIBRARY})

        add_library(ONNXRuntime SHARED IMPORTED)
        set_target_properties(ONNXRuntime
        PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIRS}"
            IMPORTED_LOCATION ${ONNXRuntime_IMPORTED_LOCATION}
            IMPORTED_IMPLIB ${ONNXRuntime_LIBRARIES}
        )
    endif()
endif()
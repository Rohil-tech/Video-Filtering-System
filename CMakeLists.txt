cmake_minimum_required(VERSION 3.15)
project(Assignment1)

set(CMAKE_CXX_STANDARD 17)

# Project layout
set(PROJECT_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(PROJECT_BINARY_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# OpenCV
find_package(OpenCV REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_DIR "C:/ONNXruntime/onnxruntime/onnxruntime-win-x64-gpu-1.17.1")

include_directories(
    ${PROJECT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

link_directories(${ONNXRUNTIME_DIR}/lib)

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR})

# Detect CUDA provider
if (EXISTS "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_cuda.dll")
    set(ONNXRUNTIME_USE_CUDA ON)
else()
    set(ONNXRUNTIME_USE_CUDA OFF)
endif()

# Library sources
set(LIBRARY_SOURCES
    ${PROJECT_SOURCE_DIR}/filter.cpp
    ${PROJECT_SOURCE_DIR}/filterApp.cpp
    ${PROJECT_SOURCE_DIR}/faceDetect.cpp
)

# Create shared library
add_library(project_lib ${LIBRARY_SOURCES})

target_link_libraries(project_lib
    ${OpenCV_LIBS}
    onnxruntime
)

# Enable CUDA only if provider exists
if (ONNXRUNTIME_USE_CUDA)
    target_compile_definitions(project_lib PRIVATE USE_CUDA=1)
endif()

# ONNX Runtime DLL list
set(ONNXRUNTIME_DLLS
    onnxruntime.dll
    onnxruntime_providers_cuda.dll
    onnxruntime_providers_shared.dll
)

# Executable helper
function(add_opencv_executable target_name)
    add_executable(${target_name}
        ${PROJECT_SOURCE_DIR}/${target_name}.cpp
    )

    target_link_libraries(${target_name}
        project_lib
        ${OpenCV_LIBS}
        onnxruntime
    )

    # Remove old ONNX Runtime DLLs
    foreach(dll ${ONNXRUNTIME_DLLS})
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rm -f
            "$<TARGET_FILE_DIR:${target_name}>/${dll}"
        )
    endforeach()

    # Copy core ONNX Runtime DLL
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:${target_name}>
    )

    # Copy CUDA provider if available
    if (ONNXRUNTIME_USE_CUDA)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_cuda.dll"
            $<TARGET_FILE_DIR:${target_name}>
        )
    endif()

    # Copy shared providers DLL if present
    if (EXISTS "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll")
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll"
            $<TARGET_FILE_DIR:${target_name}>
        )
    endif()
endfunction()

# Executables
add_opencv_executable(imgDisplay)
add_opencv_executable(vidDisplay)
add_opencv_executable(timeBlur)
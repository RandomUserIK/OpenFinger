cmake_minimum_required(VERSION 3.5)

project(OpenFinger
        VERSION 0.1.0
        LANGUAGES CXX
        DESCRIPTION "OpenFinger Finger Vein Matching Module"
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(ArrayFire REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS opencv_core
                                        opencv_imgproc
                                        opencv_imgcodecs
                                        opencv_highgui)

add_library(projectwarnings INTERFACE)

if (MSVC)
    target_compile_options(
            projectwarnings
            INTERFACE
            /W4
    )
else()
    target_compile_options(
            projectwarnings
            INTERFACE
            -Wall
            -Wextra                     # reasonable and standard
            -Wshadow                    # warn the user if a variable declaration shadows one from a parent context
            -Wnon-virtual-dtor          # warn the user if a class with virtual functions has a non-virtual destructor
            -Wold-style-cast            # warn for c-style casts
            -Wcast-align                # warn for potential performance problem casts
            -Wunused                    # warn on anything being unused
            -Woverloaded-virtual        # warn if you overload (not override) a virtual function
            -Wpedantic                  # warn if non-standard C++ is used
            -Wconversion                # warn on type conversions that may lose data
            -Wsign-conversion           # warn on sign conversions
            -Wnull-dereference          # warn if a null dereference is detected
            -Wdouble-promotion          # warn if float is implicit promoted to double
            -Wformat=2                  # warn on security issues around functions that format output
    )
endif()

set(PREPROCESSING_LIB_SOURCES
        src/matching.cpp
)
add_library(fvmatchinglib SHARED ${PREPROCESSING_LIB_SOURCES})
target_include_directories(fvmatchinglib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)

# projectwarnings
target_link_libraries(fvmatchinglib PRIVATE projectwarnings)

# Qt
target_link_libraries(fvmatchinglib PRIVATE Qt5::Core)
target_link_libraries(fvmatchinglib PRIVATE Qt5::Gui)

# OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(fvmatchinglib PRIVATE ${OpenCV_LIBS})

# ArrayFire
include_directories(${ArrayFire_INCLUDE_DIRS})
target_link_libraries(fvmatchinglib PRIVATE ${ArrayFire_LIBRARIES})

target_compile_definitions(fvmatchinglib PRIVATE MATCHING_LIBRARY)

# Set the name of the output library to be fvmatchinglib.so instead of libfvmatchinglib.so on Linux
set_target_properties(fvmatchinglib
        PROPERTIES
        PREFIX ""
        OUTPUT_NAME "fvmatchinglib"
)
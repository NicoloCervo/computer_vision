cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17) # Set the C++ standard to use.
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Set the specified C++ standard as mandatory.

project(Lab6) # Specify a project name. 

# If on Windows, set the appropriate directory for OpenCV
if (WIN32)
    set (OpenCV_DIR $ENV{OPENCV_DIR}/build)
endif ()

# Set the appropriate compiler flags.
if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /W4")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif (MSVC)

find_package(OpenCV REQUIRED)

include_directories(
    include 
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(
    ${PROJECT_NAME} 
    "src/lab6_main.cpp"
    "src/lab6.cpp"
 )

target_link_libraries(
    ${PROJECT_NAME} 
    ${OpenCV_LIBS}
)
cmake_minimum_required(VERSION 2.8.12.2)

set(OpenCV_DIR /afs/ece.cmu.edu/usr/sushmitv/Private/build/)

SET(GCC_FLAGS "-O3 -mavx -mfma")
add_definitions(${GCC_FLAGS})

project(AverageFilterProject)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Specify C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Change the target executable name here
add_executable(average_filter average_filter.cpp)

target_link_libraries(average_filter ${OpenCV_LIBS})

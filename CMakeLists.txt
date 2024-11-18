cmake_minimum_required(VERSION 3.1)

project(AverageFilterProject)

# Set OpenCV directory if needed
set(OpenCV_DIR /afs/ece.cmu.edu/usr/sushmitv/Private/build/)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Specify C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags as a list (remove quotes)
set(GCC_FLAGS -O3 -mavx -mfma)

# Add executable
add_executable(average_filter average_filter.cpp)

# Add compiler flags to the target
target_compile_options(average_filter PRIVATE ${GCC_FLAGS})

# Link libraries
target_link_libraries(average_filter PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

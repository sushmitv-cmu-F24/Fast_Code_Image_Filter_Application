# Code tested on ece027

# Fast_Code_Image_Filter_Application
C codes for Image filtering algorithms (Averaging, Gaussian and Median) which are efficient and created using multiple techniques including SIMD instructions, Kernel design, Benchmarking and more

#### Step 1: Configure path for OpenCV Build in CMakeLists.txt:
set(OpenCV_DIR /path/to/opencv/build)

#### Step 2: To execute the average_filter.cpp, gaussian_filter.cpp and median_filter.cpp files follow below steps
```
1. cd build/
2. cmake ../
3. make
4. ./average_filter
5. ./gaussian_filter
6. ./median_filter
```
#### Check avg_kernel_output_images and gaussian_ker_output_images for output images
#### Check average_performance_data and gaussian_performance_data for performance results


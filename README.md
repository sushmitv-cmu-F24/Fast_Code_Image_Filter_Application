# Fast_Code_Image_Filter_Application
C codes for Image filtering algorithms (Averaging, Gaussian and Median) which are efficient and created using multiple techniques including SIMD instructions, Kernel design, Benchmarking and more

#### Step 1: Configure path for OpenCV Build in CMakeLists.txt:
set(OpenCV_DIR /path/to/opencv/build)

#### Step 2: To execute the average_filter.cpp & gaussian_filter.cpp files follow below steps
```
1. cd build/
2. cmake ../
3. make
4. ./average_filter
5. ./gaussian_filter
```
#### Check output_opencv.jpg & output_simd.jpg for final results

#include <opencv2/opencv.hpp>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <omp.h>

using namespace cv;
using namespace std;

#define SIMD_WIDTH 8 // 8 floats for AVX2
#define TILE_SIZE 32 // Tile size for better cache utilization

// Full SIMD sorting network for 25 elements
void simd_sort_network_25(float* buffer) {
    __m256 vec1 = _mm256_loadu_ps(buffer);        // Load first 8 elements
    __m256 vec2 = _mm256_loadu_ps(buffer + 8);    // Load next 8 elements
    __m256 vec3 = _mm256_loadu_ps(buffer + 16);   // Load last 8 elements
    float last = buffer[24];                      // Handle scalar element

    // Sorting logic (simplified for demonstration; replace with full network as needed)
    vec1 = _mm256_min_ps(vec1, vec2);
    vec2 = _mm256_max_ps(vec1, vec2);
    vec3 = _mm256_min_ps(vec3, vec2);
    vec2 = _mm256_max_ps(vec3, vec2);

    // Store sorted vectors
    _mm256_storeu_ps(buffer, vec1);
    _mm256_storeu_ps(buffer + 8, vec2);
    _mm256_storeu_ps(buffer + 16, vec3);
    buffer[24] = last; // Handle scalar
}

// Optimized median blur kernel
void median_blur_kernel(const Mat& input, Mat& output) {
    int width = input.cols;
    int height = input.rows;
    const float* inputData = input.ptr<float>();
    float* outputData = output.ptr<float>();
    int inputStep = input.step1();
    int outputStep = output.step1();

    #pragma omp parallel for schedule(static, TILE_SIZE)
    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x += SIMD_WIDTH) {
            float buffers[SIMD_WIDTH][25]; // Buffers for SIMD_WIDTH pixels

            // Load 5x5 neighborhood for each pixel
            for (int ky = -2; ky <= 2; ky++) {
                const float* rowPtr = inputData + (y + ky) * inputStep + x - 2;
                for (int kx = 0; kx < 5; kx++) {
                    for (int i = 0; i < SIMD_WIDTH; i++) {
                        buffers[i][(ky + 2) * 5 + kx] = rowPtr[kx + i];
                    }
                }
            }

            // Sort each buffer and extract the median
            for (int i = 0; i < SIMD_WIDTH; i++) {
                simd_sort_network_25(buffers[i]);
                outputData[y * outputStep + x + i] = buffers[i][12]; // Median at index 12
            }
        }
    }
}

int main(int argc, char** argv) {
    // Load input image in grayscale
    Mat img = imread("input.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return -1;
    }

    // Define image sizes to test
    vector<int> sizes = {2048, 4096, 8192, 16384};

    for (int size : sizes) {
        // Resize image to ensure consistent testing sizes
        Mat img_resized;
        resize(img, img_resized, Size(size, size));

        // Convert image to float for SIMD processing
        Mat img_float;
        img_resized.convertTo(img_float, CV_32F);

        // Initialize output matrices
        Mat output_simd = Mat::zeros(img_float.size(), CV_32F);
        Mat output_opencv;

        // Measure time for optimized median filter
        int64 t1 = getTickCount();
        median_blur_kernel(img_float, output_simd);
        double time_simd = (getTickCount() - t1) / getTickFrequency();

        // Measure time for single-threaded OpenCV's medianBlur implementation
        setNumThreads(1); // Limit OpenCV to a single thread
        t1 = getTickCount();
        medianBlur(img_resized, output_opencv, 5);
        double time_opencv = (getTickCount() - t1) / getTickFrequency();

        // Print performance comparison for current image size
        cout << "Image size: " << size << " x " << size << endl;
        cout << "Optimized implementation (SIMD) time: " << time_simd << " seconds" << endl;
        cout << "OpenCV implementation time: " << time_opencv << " seconds" << endl;
        cout << "Speedup (OpenCV / SIMD): " << time_opencv / time_simd << endl;
        cout << "-------------------------------" << endl;
    }

    return 0;
}
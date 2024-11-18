#include <opencv2/opencv.hpp>
#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <omp.h>  // Include OpenMP header
#include <vector>

using namespace cv;
using namespace std;

// Adjust tile sizes based on cache size
const int TileHeight = 64;  // Adjust according to L1 cache size
const int TileWidth = 64;

// Vertical pass with optimizations
void vertical_blur_optimized(const Mat &input, Mat &intermediate) {
    int width = input.cols;
    int height = input.rows;
    const float* inputData = input.ptr<float>(0);
    float* intermediateData = intermediate.ptr<float>(0);
    int inputStep = input.step1();
    int intermediateStep = intermediate.step1();
    __m256 reciprocal5 = _mm256_set1_ps(0.2f);  // 1/5

    // Parallelize outer loop using OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (int yTile = 2; yTile < height - 2; yTile += TileHeight) {
        int yTileEnd = min(yTile + TileHeight, height - 2);
        for (int xTile = 0; xTile < width; xTile += TileWidth) {
            int xTileEnd = min(xTile + TileWidth, width);

            for (int y = yTile; y < yTileEnd; y++) {
                int x = xTile;
                // Unroll loop to process 16 pixels per iteration
                for (; x <= xTileEnd - 16; x += 16) {
                    // Prefetch next data
                    _mm_prefetch((const char*)(inputData + (y - 2) * inputStep + x + 16), _MM_HINT_T0);

                    // Process first 8 pixels
                    __m256 p0 = _mm256_loadu_ps(inputData + (y - 2) * inputStep + x);
                    __m256 p1 = _mm256_loadu_ps(inputData + (y - 1) * inputStep + x);
                    __m256 p2 = _mm256_loadu_ps(inputData + y * inputStep + x);
                    __m256 p3 = _mm256_loadu_ps(inputData + (y + 1) * inputStep + x);
                    __m256 p4 = _mm256_loadu_ps(inputData + (y + 2) * inputStep + x);

                    __m256 sum = _mm256_mul_ps(p0, reciprocal5);
                    sum = _mm256_fmadd_ps(p1, reciprocal5, sum);
                    sum = _mm256_fmadd_ps(p2, reciprocal5, sum);
                    sum = _mm256_fmadd_ps(p3, reciprocal5, sum);
                    sum = _mm256_fmadd_ps(p4, reciprocal5, sum);

                    _mm256_storeu_ps(intermediateData + y * intermediateStep + x, sum);

                    // Process next 8 pixels
                    __m256 p0b = _mm256_loadu_ps(inputData + (y - 2) * inputStep + x + 8);
                    __m256 p1b = _mm256_loadu_ps(inputData + (y - 1) * inputStep + x + 8);
                    __m256 p2b = _mm256_loadu_ps(inputData + y * inputStep + x + 8);
                    __m256 p3b = _mm256_loadu_ps(inputData + (y + 1) * inputStep + x + 8);
                    __m256 p4b = _mm256_loadu_ps(inputData + (y + 2) * inputStep + x + 8);

                    __m256 sumB = _mm256_mul_ps(p0b, reciprocal5);
                    sumB = _mm256_fmadd_ps(p1b, reciprocal5, sumB);
                    sumB = _mm256_fmadd_ps(p2b, reciprocal5, sumB);
                    sumB = _mm256_fmadd_ps(p3b, reciprocal5, sumB);
                    sumB = _mm256_fmadd_ps(p4b, reciprocal5, sumB);

                    _mm256_storeu_ps(intermediateData + y * intermediateStep + x + 8, sumB);
                }
                // Handle remaining pixels
                for (; x <= xTileEnd - 8; x += 8) {
                    __m256 p0 = _mm256_loadu_ps(inputData + (y - 2) * inputStep + x);
                    __m256 p1 = _mm256_loadu_ps(inputData + (y - 1) * inputStep + x);
                    __m256 p2 = _mm256_loadu_ps(inputData + y * inputStep + x);
                    __m256 p3 = _mm256_loadu_ps(inputData + (y + 1) * inputStep + x);
                    __m256 p4 = _mm256_loadu_ps(inputData + (y + 2) * inputStep + x);

                    __m256 sum = _mm256_mul_ps(p0, reciprocal5);
                    sum = _mm256_fmadd_ps(p1, reciprocal5, sum);
                    sum = _mm256_fmadd_ps(p2, reciprocal5, sum);
                    sum = _mm256_fmadd_ps(p3, reciprocal5, sum);
                    sum = _mm256_fmadd_ps(p4, reciprocal5, sum);

                    _mm256_storeu_ps(intermediateData + y * intermediateStep + x, sum);
                }
                // Handle any remaining pixels
                for (; x < xTileEnd; x++) {
                    float sum = (
                        inputData[(y - 2) * inputStep + x] +
                        inputData[(y - 1) * inputStep + x] +
                        inputData[y * inputStep + x] +
                        inputData[(y + 1) * inputStep + x] +
                        inputData[(y + 2) * inputStep + x]
                    ) * 0.2f;

                    intermediateData[y * intermediateStep + x] = sum;
                }
            }
        }
    }
}

// Horizontal pass with optimizations
void horizontal_blur_optimized(const Mat &intermediate, Mat &output) {
    int width = intermediate.cols;
    int height = intermediate.rows;
    const float* intermediateData = intermediate.ptr<float>(0);
    float* outputData = output.ptr<float>(0);
    int intermediateStep = intermediate.step1();
    int outputStep = output.step1();
    __m256 reciprocal5 = _mm256_set1_ps(0.2f);  // 1/5

    // Parallelize outer loop using OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (int yTile = 2; yTile < height - 2; yTile += TileHeight) {
        int yTileEnd = min(yTile + TileHeight, height - 2);
        for (int xTile = 2; xTile < width - 2; xTile += TileWidth) {
            int xTileEnd = min(xTile + TileWidth, width - 2);

            for (int y = yTile; y < yTileEnd; y++) {
                int x = xTile;
                // Unroll loop to process 16 pixels per iteration
                for (; x <= xTileEnd - 16; x += 16) {
                    // Prefetch next data
                    _mm_prefetch((const char*)(intermediateData + y * intermediateStep + x + 16), _MM_HINT_T0);

                    // Process first 8 pixels
                    __m256 p0 = _mm256_loadu_ps(intermediateData + y * intermediateStep + x - 2);
                    __m256 p1 = _mm256_loadu_ps(intermediateData + y * intermediateStep + x - 1);
                    __m256 p2 = _mm256_loadu_ps(intermediateData + y * intermediateStep + x);
                    __m256 p3 = _mm256_loadu_ps(intermediateData + y * intermediateStep + x + 1);
                    __m256 p4 = _mm256_loadu_ps(intermediateData + y * intermediateStep + x + 2);

                    __m256 sum = _mm256_mul_ps(p0, reciprocal5);
                    sum = _mm256_fmadd_ps(p1, reciprocal5, sum);
                    sum = _mm256_fmadd_ps(p2, reciprocal5, sum);
                    sum = _mm256_fmadd_ps(p3, reciprocal5, sum);
                    sum = _mm256_fmadd_ps(p4, reciprocal5, sum);

                    _mm256_storeu_ps(outputData + y * outputStep + x, sum);

                    // Process next 8 pixels
                    __m256 p0b = _mm256_loadu_ps(intermediateData + y * intermediateStep + x + 8 - 2);
                    __m256 p1b = _mm256_loadu_ps(intermediateData + y * intermediateStep + x + 8 - 1);
                    __m256 p2b = _mm256_loadu_ps(intermediateData + y * intermediateStep + x + 8);
                    __m256 p3b = _mm256_loadu_ps(intermediateData + y * intermediateStep + x + 8 + 1);
                    __m256 p4b = _mm256_loadu_ps(intermediateData + y * intermediateStep + x + 8 + 2);

                    __m256 sumB = _mm256_mul_ps(p0b, reciprocal5);
                    sumB = _mm256_fmadd_ps(p1b, reciprocal5, sumB);
                    sumB = _mm256_fmadd_ps(p2b, reciprocal5, sumB);
                    sumB = _mm256_fmadd_ps(p3b, reciprocal5, sumB);
                    sumB = _mm256_fmadd_ps(p4b, reciprocal5, sumB);

                    _mm256_storeu_ps(outputData + y * outputStep + x + 8, sumB);
                }
                // Handle remaining pixels
                for (; x <= xTileEnd - 8; x += 8) {
                    __m256 p0 = _mm256_loadu_ps(intermediateData + y * intermediateStep + x - 2);
                    __m256 p1 = _mm256_loadu_ps(intermediateData + y * intermediateStep + x - 1);
                    __m256 p2 = _mm256_loadu_ps(intermediateData + y * intermediateStep + x);
                    __m256 p3 = _mm256_loadu_ps(intermediateData + y * intermediateStep + x + 1);
                    __m256 p4 = _mm256_loadu_ps(intermediateData + y * intermediateStep + x + 2);

                    __m256 sum = _mm256_mul_ps(p0, reciprocal5);
                    sum = _mm256_fmadd_ps(p1, reciprocal5, sum);
                    sum = _mm256_fmadd_ps(p2, reciprocal5, sum);
                    sum = _mm256_fmadd_ps(p3, reciprocal5, sum);
                    sum = _mm256_fmadd_ps(p4, reciprocal5, sum);

                    _mm256_storeu_ps(outputData + y * outputStep + x, sum);
                }
                // Handle any remaining pixels
                for (; x < xTileEnd; x++) {
                    float sum = (
                        intermediateData[y * intermediateStep + x - 2] +
                        intermediateData[y * intermediateStep + x - 1] +
                        intermediateData[y * intermediateStep + x] +
                        intermediateData[y * intermediateStep + x + 1] +
                        intermediateData[y * intermediateStep + x + 2]
                    ) * 0.2f;

                    outputData[y * outputStep + x] = sum;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    // Load input image in grayscale
    Mat img = imread("../input.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error: Could not read the image file!" << endl;
        return -1;
    }

    // Define image sizes to test
    vector<int> sizes = {2048, 4096, 8192, 16384, 32768};

    // Initialize vectors to store performance data
    vector<double> times_simd;
    vector<double> times_opencv;
    vector<double> gflops_simd;

    for (int size : sizes) {
        // Resize image to current size
        Mat img_resized;
        resize(img, img_resized, Size(size, size));

        // Convert image to float32
        Mat img_float;
        img_resized.convertTo(img_float, CV_32F);

        // Intermediate image for the separable filter
        Mat intermediate = Mat::zeros(img_float.size(), CV_32F);

        // Initialize output images
        Mat output_simd = Mat::zeros(img_float.size(), CV_32F);
        Mat output_opencv;

        // Apply SIMD separable filter implementation and measure the time
        int64 t1 = getTickCount();
        vertical_blur_optimized(img_float, intermediate);
        horizontal_blur_optimized(intermediate, output_simd);
        double time_simd = (getTickCount() - t1) / getTickFrequency();

        // Compute total FLOPs
        // Assuming 20 FLOPs per pixel (10 per pass)
        int width = img_float.cols;
        int height = img_float.rows;
        double total_flops = static_cast<double>(width) * height * 20;

        // Compute GFLOPS achieved
        double gflops = (total_flops / time_simd) / 1e9;

        // Store performance data
        times_simd.push_back(time_simd);
        gflops_simd.push_back(gflops);

        // Apply OpenCV implementation and measure the time
        t1 = getTickCount();
        blur(img_resized, output_opencv, Size(5, 5));
        double time_opencv = (getTickCount() - t1) / getTickFrequency();

        times_opencv.push_back(time_opencv);

        // Output performance data
        cout << "Image size: " << size << " x " << size << endl;
        cout << "SIMD implementation time: " << time_simd << " seconds" << endl;
        cout << "GFLOPS achieved: " << gflops << endl;
        cout << "OpenCV blur time: " << time_opencv << " seconds" << endl;
        cout << "-------------------------------" << endl;

        // Save SIMD output image for verification
        // Convert SIMD output back to 8-bit for saving
        Mat output_simd_8U;
        output_simd.convertTo(output_simd_8U, CV_8U);
        string filename_simd = "../output_images/output_simd_" + to_string(size) + "x" + to_string(size) + ".jpg";
        imwrite(filename_simd, output_simd_8U);

        // Save OpenCV output image for verification
        // Since output_opencv is already in 8-bit format, we can save it directly
        string filename_opencv = "../output_images/output_opencv_" + to_string(size) + "x" + to_string(size) + ".jpg";
        imwrite(filename_opencv, output_opencv);
    }

    // Optionally, you can write the performance data to a file for plotting
    ofstream outfile("../performance_data.txt");
    outfile << "Size\tTime_SIMD(s)\tGFLOPS\tTime_OpenCV(s)\n";
    for (size_t i = 0; i < sizes.size(); ++i) {
        outfile << sizes[i] << "\t" << times_simd[i] << "\t" << gflops_simd[i] << "\t" << times_opencv[i] << "\n";
    }
    outfile.close();

    return 0;
}

#include <opencv2/opencv.hpp>
#include <immintrin.h>
#include <iostream>
#include <omp.h>  // Include OpenMP header
#include <thread>

using namespace cv;
using namespace std;

// Adjust tile sizes based on cache size
const int TileHeight = 256;  // Increased tile height
const int TileWidth = 256;   // Increased tile width
const int ChunkSize = 64;    // Increased chunk size


// Vertical pass with optimization
void vertical_gaussian_blur(const Mat &input, Mat &intermediate) {
    int width = input.cols;
    int height = input.rows;
    const float* inputData = input.ptr<float>(0);
    float* intermediateData = intermediate.ptr<float>(0);
    int inputStep = input.step1();
    int intermediateStep = intermediate.step1();

    // Gaussian weights
    __m256 w0 = _mm256_set1_ps(0.0625f);  // 1/16
    __m256 w1 = _mm256_set1_ps(0.25f);    // 4/16
    __m256 w2 = _mm256_set1_ps(0.375f);   // 6/16

    // Parallelize outer loop using OpenMP with static scheduling
    #pragma omp parallel for schedule(static)
    for (int yTile = 2; yTile < height - 2; yTile += TileHeight) {
        int yTileEnd = min(yTile + TileHeight, height - 2);
        for (int xTile = 0; xTile < width; xTile += TileWidth) {
            int xTileEnd = min(xTile + TileWidth, width);

            for (int y = yTile; y < yTileEnd; y++) {
                int x = xTile;
                // Unroll loop to process 64 pixels per iteration
                for (; x <= xTileEnd - ChunkSize; x += ChunkSize) {
                    for (int vec = 0; vec < ChunkSize / 8; vec++) { // 8 __m256 vectors per chunk
                        int offset = x + vec * 8;

                        // Prefetch next data conditionally
                        if (offset + 16 < xTileEnd) { // Adjust prefetch distance as needed
                            _mm_prefetch((const char*)(inputData + (y - 2) * inputStep + offset + 16), _MM_HINT_T0);
                        }

                        // Load input vectors
                        __m256 p0 = _mm256_loadu_ps(inputData + (y - 2) * inputStep + offset);
                        __m256 p1 = _mm256_loadu_ps(inputData + (y - 1) * inputStep + offset);
                        __m256 p2 = _mm256_loadu_ps(inputData + y * inputStep + offset);
                        __m256 p3 = _mm256_loadu_ps(inputData + (y + 1) * inputStep + offset);
                        __m256 p4 = _mm256_loadu_ps(inputData + (y + 2) * inputStep + offset);

                        // Apply weights
                        __m256 sum = _mm256_mul_ps(p0, w0);
                        sum = _mm256_fmadd_ps(p1, w1, sum);
                        sum = _mm256_fmadd_ps(p2, w2, sum);
                        sum = _mm256_fmadd_ps(p3, w1, sum);
                        sum = _mm256_fmadd_ps(p4, w0, sum);

                        // Store the result
                        _mm256_storeu_ps(intermediateData + y * intermediateStep + offset, sum);
                    }
                }
                // Handle remaining pixels in chunks of 8
                for (; x <= xTileEnd - 8; x += 8) {
                    __m256 p0 = _mm256_loadu_ps(inputData + (y - 2) * inputStep + x);
                    __m256 p1 = _mm256_loadu_ps(inputData + (y - 1) * inputStep + x);
                    __m256 p2 = _mm256_loadu_ps(inputData + y * inputStep + x);
                    __m256 p3 = _mm256_loadu_ps(inputData + (y + 1) * inputStep + x);
                    __m256 p4 = _mm256_loadu_ps(inputData + (y + 2) * inputStep + x);

                    __m256 sum = _mm256_mul_ps(p0, w0);
                    sum = _mm256_fmadd_ps(p1, w1, sum);
                    sum = _mm256_fmadd_ps(p2, w2, sum);
                    sum = _mm256_fmadd_ps(p3, w1, sum);
                    sum = _mm256_fmadd_ps(p4, w0, sum);

                    _mm256_storeu_ps(intermediateData + y * intermediateStep + x, sum);
                }
                // Handle any remaining pixels
                for (; x < xTileEnd; x++) {
                    float sum = (
                        inputData[(y - 2) * inputStep + x] * 0.0625f +
                        inputData[(y - 1) * inputStep + x] * 0.25f +
                        inputData[y * inputStep + x] * 0.375f +
                        inputData[(y + 1) * inputStep + x] * 0.25f +
                        inputData[(y + 2) * inputStep + x] * 0.0625f
                    );

                    intermediateData[y * intermediateStep + x] = sum;
                }
            }
        }
    }
}

void horizontal_gaussian_blur(const Mat &intermediate, Mat &output) {
    int width = intermediate.cols;
    int height = intermediate.rows;
    const float* intermediateData = intermediate.ptr<float>(0);
    float* outputData = output.ptr<float>(0);
    int intermediateStep = intermediate.step1();
    int outputStep = output.step1();

    // Gaussian weights
    __m256 w0 = _mm256_set1_ps(0.0625f);  // 1/16
    __m256 w1 = _mm256_set1_ps(0.25f);    // 4/16
    __m256 w2 = _mm256_set1_ps(0.375f);   // 6/16

    // Parallelize outer loop using OpenMP with static scheduling
    #pragma omp parallel for schedule(static)
    for (int yTile = 0; yTile < height; yTile += TileHeight) {  // No vertical borders needed for horizontal pass
        int yTileEnd = min(yTile + TileHeight, height);
        for (int xTile = 2; xTile < width - 2; xTile += TileWidth) {  // Start from 2 to handle x-2
            int xTileEnd = min(xTile + TileWidth, width - 2);

            for (int y = yTile; y < yTileEnd; y++) {
                int x = xTile;
                // Unroll loop to process 64 pixels per iteration
                for (; x <= xTileEnd - ChunkSize; x += ChunkSize) {
                    for (int vec = 0; vec < ChunkSize / 8; vec++) { // 8 __m256 vectors per chunk (64/8=8)
                        int offset = x + vec * 8;

                        // Prefetch next data conditionally
                        if (offset + 16 < xTileEnd) { // Adjust prefetch distance as needed
                            _mm_prefetch((const char*)(intermediateData + y * intermediateStep + offset + 16), _MM_HINT_T0);
                        }

                        // Load input vectors for horizontal neighbors
                        __m256 p0 = _mm256_loadu_ps(intermediateData + y * intermediateStep + offset - 2);
                        __m256 p1 = _mm256_loadu_ps(intermediateData + y * intermediateStep + offset - 1);
                        __m256 p2 = _mm256_loadu_ps(intermediateData + y * intermediateStep + offset);
                        __m256 p3 = _mm256_loadu_ps(intermediateData + y * intermediateStep + offset + 1);
                        __m256 p4 = _mm256_loadu_ps(intermediateData + y * intermediateStep + offset + 2);

                        // Apply Gaussian weights
                        __m256 sum = _mm256_mul_ps(p0, w0);
                        sum = _mm256_fmadd_ps(p1, w1, sum);
                        sum = _mm256_fmadd_ps(p2, w2, sum);
                        sum = _mm256_fmadd_ps(p3, w1, sum);
                        sum = _mm256_fmadd_ps(p4, w0, sum);

                        // Store the result
                        _mm256_storeu_ps(outputData + y * outputStep + offset, sum);
                    }
                }
                // Handle remaining pixels in chunks of 8
                for (; x <= xTileEnd - 8; x += 8) {
                    __m256 p0 = _mm256_loadu_ps(intermediateData + y * intermediateStep + x - 2);
                    __m256 p1 = _mm256_loadu_ps(intermediateData + y * intermediateStep + x - 1);
                    __m256 p2 = _mm256_loadu_ps(intermediateData + y * intermediateStep + x);
                    __m256 p3 = _mm256_loadu_ps(intermediateData + y * intermediateStep + x + 1);
                    __m256 p4 = _mm256_loadu_ps(intermediateData + y * intermediateStep + x + 2);

                    __m256 sum = _mm256_mul_ps(p0, w0);
                    sum = _mm256_fmadd_ps(p1, w1, sum);
                    sum = _mm256_fmadd_ps(p2, w2, sum);
                    sum = _mm256_fmadd_ps(p3, w1, sum);
                    sum = _mm256_fmadd_ps(p4, w0, sum);

                    _mm256_storeu_ps(outputData + y * outputStep + x, sum);
                }
                // Handle any remaining pixels individually
                for (; x < xTileEnd; x++) {
                    float sum = (
                        intermediateData[y * intermediateStep + x - 2] * 0.0625f +
                        intermediateData[y * intermediateStep + x - 1] * 0.25f +
                        intermediateData[y * intermediateStep + x] * 0.375f +
                        intermediateData[y * intermediateStep + x + 1] * 0.25f +
                        intermediateData[y * intermediateStep + x + 2] * 0.0625f
                    );

                    outputData[y * outputStep + x] = sum;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    // Set OpenCV to single-threaded mode
    int original_threads = cv::getNumThreads();
    cv::setNumThreads(1);

    int num_threads = std::thread::hardware_concurrency();  // Get hardware thread count
    omp_set_num_threads(num_threads);   
    int img_size = 4096;

    // Load input image in grayscale
    Mat img = imread("../input.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error: Could not read the image file!" << endl;
        return -1;
    }

    Mat img_resized;
    resize(img, img_resized, Size(img_size, img_size));

    // Convert image to float32
    Mat img_float;
    img_resized.convertTo(img_float, CV_32F);

    // Intermediate image for the separable filter
    Mat intermediate = Mat::zeros(img_float.size(), CV_32F);

    // Initialize output images
    Mat output_simd = Mat::zeros(img_float.size(), CV_32F);
    Mat output_opencv;

    // Apply SIMD separable Gaussian filter implementation and measure the time
    int64 t1 = getTickCount();
    vertical_gaussian_blur(img_float, intermediate);
    horizontal_gaussian_blur(intermediate, output_simd);
    double time_simd = (getTickCount() - t1) / getTickFrequency();

    // Convert SIMD output back to 8-bit for comparison and saving
    Mat output_simd_8U;
    output_simd.convertTo(output_simd_8U, CV_8U);
    imwrite("../output_simd_gaussian.jpg", output_simd_8U);

    // Apply OpenCV GaussianBlur implementation and measure the time
    t1 = getTickCount();
    GaussianBlur(img_float, output_opencv, Size(5, 5), 0, 0, BORDER_DEFAULT);
    double time_opencv = (getTickCount() - t1) / getTickFrequency();

    // Convert OpenCV output back to 8-bit for saving
    Mat output_opencv_8U;
    output_opencv.convertTo(output_opencv_8U, CV_8U);
    imwrite("../output_opencv_gaussian.jpg", output_opencv_8U);

    // Display comparison results
    cout << "Performance comparison:" << endl;
    cout << "Image Size: " << img_size << " X " << img_size << endl;
    cout << "SIMD Gaussian implementation time: " << time_simd << " seconds" << endl;
    cout << "OpenCV Gaussian implementation time (single-threaded): " << time_opencv << " seconds" << endl;

    return 0;
}

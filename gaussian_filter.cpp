#include <opencv2/opencv.hpp>
#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <omp.h>  // Include OpenMP header
#include <thread>

using namespace cv;
using namespace std;

static const double CPU_FREQ_GHZ = 3.2;

// Constants for tiling and chunking
const int TileHeight = 256;  // Increased tile height based on cache analysis
const int TileWidth = 256;   // Increased tile width based on cache analysis
const int ChunkSize = 64;    // Increased chunk size to process 64 pixels per iteration

// Vertical pass with optimizations and loop unrolling
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
                // Unroll loop to process 64 pixels per iteration with a factor of 4
                for (; x <= xTileEnd - ChunkSize; x += ChunkSize) {
                    // Unrolled loop: processing 4 vectors (32 pixels) per sub-iteration
                    for (int sub = 0; sub < ChunkSize / 32; sub++) { // 64 / 32 = 2 sub-iterations
                        for (int vec = 0; vec < 4; vec++) { // 4 vectors per sub-iteration
                            int offset = x + sub * 32 + vec * 8;

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
                }
                // Handle remaining pixels in chunks of 32
                for (; x <= xTileEnd - 32; x += 32) {
                    for (int vec = 0; vec < 4; vec++) { // 4 vectors per 32-pixel chunk
                        int offset = x + vec * 8;

                        // Prefetch next data conditionally
                        if (offset + 16 < xTileEnd) {
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
                // Handle any remaining pixels individually
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

// Horizontal pass with optimizations and loop unrolling
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
                // Unroll loop to process 64 pixels per iteration with a factor of 4
                for (; x <= xTileEnd - ChunkSize; x += ChunkSize) {
                    // Unrolled loop: processing 4 vectors (32 pixels) per sub-iteration
                    for (int sub = 0; sub < ChunkSize / 32; sub++) { // 64 / 32 = 2 sub-iterations
                        for (int vec = 0; vec < 4; vec++) { // 4 vectors per sub-iteration
                            int offset = x + sub * 32 + vec * 8;

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
                }
                // Handle remaining pixels in chunks of 32
                for (; x <= xTileEnd - 32; x += 32) {
                    for (int vec = 0; vec < 4; vec++) { // 4 vectors per 32-pixel chunk
                        int offset = x + vec * 8;

                        // Prefetch next data conditionally
                        if (offset + 16 < xTileEnd) {
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
    // Disable OpenCV's internal optimizations and set to single-threaded mode for fair comparison
    cv::setUseOptimized(false);
    cv::setNumThreads(1);

    // Load input image in grayscale
    Mat img = imread("../input.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error: Could not read the image file!" << endl;
        return -1;
    }

    // Define the image sizes to test
    vector<int> sizes = {2048, 4096, 8192};
    // Vectors to store performance metrics
    vector<double> times_simd;
    vector<double> times_opencv;
    vector<double> gflops_per_cycle;

    // Open output file to log performance data
    ofstream outfile("../gaussian_performance_data.txt");
    if (!outfile.is_open()) {
        cerr << "Error: Could not open performance_data.txt for writing!" << endl;
        return -1;
    }
    // Write header to the file
    outfile << "Size\tTime_SIMD(s)\tGFLOPs/cycle\tTime_OpenCV(s)\n";

    // Iterate over each image size
    for (int size : sizes) {
        // Resize the image to the current size
        Mat img_resized;
        resize(img, img_resized, Size(size, size));

        // Convert image to float32 for processing
        Mat img_float;
        img_resized.convertTo(img_float, CV_32F);

        // Initialize intermediate and output matrices
        Mat intermediate = Mat::zeros(img_float.size(), CV_32F);
        Mat output_simd = Mat::zeros(img_float.size(), CV_32F);
        Mat output_opencv;

        // Measure time for SIMD-optimized Gaussian blur
        int64 t1 = getTickCount();
        vertical_gaussian_blur(img_float, intermediate);
        horizontal_gaussian_blur(intermediate, output_simd);
        double time_simd = (getTickCount() - t1) / getTickFrequency();

        // Calculate total FLOPs for Gaussian blur
        // For Gaussian blur with a 5x5 kernel, each output pixel involves 20 floating-point operations
        // (5 multiplications and 4 additions per row, multiplied by 5 rows). Adjust if your kernel differs.
        double total_flops = static_cast<double>(size) * size * 20.0;

        // Calculate GFLOPs per second
        double gflops_per_sec = (total_flops / time_simd) / 1e9;

        // Calculate GFLOPs per cycle
        double gflops_per_cyc = gflops_per_sec / CPU_FREQ_GHZ;

        // Store the metrics
        times_simd.push_back(time_simd);
        gflops_per_cycle.push_back(gflops_per_cyc);

        // Measure time for OpenCV's GaussianBlur
        t1 = getTickCount();
        GaussianBlur(img_float, output_opencv, Size(5, 5), 0, 0, BORDER_DEFAULT);
        double time_opencv = (getTickCount() - t1) / getTickFrequency();
        times_opencv.push_back(time_opencv);

        // Log the results to the console
        cout << "Image size: " << size << " x " << size << endl;
        cout << "SIMD Gaussian implementation time: " << time_simd << " s" << endl;
        cout << "GFLOPs/cycle: " << gflops_per_cyc << endl;
        cout << "OpenCV GaussianBlur time: " << time_opencv << " s" << endl;
        cout << "-------------------------------" << endl;

        // Convert SIMD output back to 8-bit for saving
        Mat output_simd_8U;
        output_simd.convertTo(output_simd_8U, CV_8U);
        imwrite("../gaussian_ker_output_images/output_simd_" + to_string(size) + ".jpg", output_simd_8U);

        // Convert OpenCV output back to 8-bit for saving (if not already)
        Mat output_opencv_8U;
        output_opencv.convertTo(output_opencv_8U, CV_8U);
        imwrite("../gaussian_ker_output_images/output_opencv_" + to_string(size) + ".jpg", output_opencv_8U);

        // Write the metrics to the file
        outfile << size << "\t" << time_simd << "\t" << gflops_per_cyc << "\t" << time_opencv << "\n";
    }

    // Close the performance data file
    outfile.close();

    return 0;
}

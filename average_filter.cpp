#include <opencv2/opencv.hpp>
#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

static const double CPU_FREQ_GHZ = 3.2; // Adjust this to match your CPU frequency

const int TileHeight = 64;  
const int TileWidth = 64;

// We'll process 64 pixels at once: 8 registers of 8 floats each (8 * 8 = 64)
static const int VECTOR_WIDTH = 8;
static const int CHUNK = 64; // 64 pixels per iteration

// Vertical blur
void vertical_blur_optimized(const Mat &input, Mat &intermediate) {
    int width = input.cols;
    int height = input.rows;
    const float* inputData = input.ptr<float>(0);
    float* intermediateData = intermediate.ptr<float>(0);
    int inputStep = (int)input.step1();
    int intermediateStep = (int)intermediate.step1();

    __m256 w = _mm256_set1_ps(0.2f);

    #pragma omp parallel for schedule(dynamic)
    for (int yTile = 2; yTile < height - 2; yTile += TileHeight) {
        int yTileEnd = std::min(yTile + TileHeight, height - 2);
        for (int xTile = 0; xTile < width; xTile += TileWidth) {
            int xTileEnd = std::min(xTile + TileWidth, width);

            for (int y = yTile; y < yTileEnd; y++) {
                int x = xTile;
                // Process in chunks of 64 pixels
                for (; x <= xTileEnd - CHUNK; x += CHUNK) {
                    __m256 p0[CHUNK/VECTOR_WIDTH], p1[CHUNK/VECTOR_WIDTH], p2[CHUNK/VECTOR_WIDTH], p3[CHUNK/VECTOR_WIDTH], p4[CHUNK/VECTOR_WIDTH];

                    const float* rowm2 = inputData + (y - 2)*inputStep + x;
                    const float* rowm1 = inputData + (y - 1)*inputStep + x;
                    const float* row0  = inputData + y*inputStep + x;
                    const float* rowp1 = inputData + (y + 1)*inputStep + x;
                    const float* rowp2 = inputData + (y + 2)*inputStep + x;

                    for (int r = 0; r < CHUNK/VECTOR_WIDTH; r++) {
                        p0[r] = _mm256_loadu_ps(rowm2 + r*VECTOR_WIDTH);
                        p1[r] = _mm256_loadu_ps(rowm1 + r*VECTOR_WIDTH);
                        p2[r] = _mm256_loadu_ps(row0  + r*VECTOR_WIDTH);
                        p3[r] = _mm256_loadu_ps(rowp1 + r*VECTOR_WIDTH);
                        p4[r] = _mm256_loadu_ps(rowp2 + r*VECTOR_WIDTH);
                    }

                    for (int r = 0; r < CHUNK/VECTOR_WIDTH; r++) {
                        __m256 sum = _mm256_mul_ps(p0[r], w);
                        sum = _mm256_fmadd_ps(p1[r], w, sum);
                        sum = _mm256_fmadd_ps(p2[r], w, sum);
                        sum = _mm256_fmadd_ps(p3[r], w, sum);
                        sum = _mm256_fmadd_ps(p4[r], w, sum);
                        _mm256_storeu_ps(intermediateData + y*intermediateStep + x + r*VECTOR_WIDTH, sum);
                    }
                }

                // Handle remainder in chunks of 8
                for (; x <= xTileEnd - 8; x += 8) {
                    __m256 p0 = _mm256_loadu_ps(inputData + (y-2)*inputStep + x);
                    __m256 p1 = _mm256_loadu_ps(inputData + (y-1)*inputStep + x);
                    __m256 p2 = _mm256_loadu_ps(inputData + y*inputStep + x);
                    __m256 p3 = _mm256_loadu_ps(inputData + (y+1)*inputStep + x);
                    __m256 p4 = _mm256_loadu_ps(inputData + (y+2)*inputStep + x);

                    __m256 sum = _mm256_mul_ps(p0, w);
                    sum = _mm256_fmadd_ps(p1, w, sum);
                    sum = _mm256_fmadd_ps(p2, w, sum);
                    sum = _mm256_fmadd_ps(p3, w, sum);
                    sum = _mm256_fmadd_ps(p4, w, sum);

                    _mm256_storeu_ps(intermediateData + y*intermediateStep + x, sum);
                }

                // Scalar leftover
                for (; x < xTileEnd; x++) {
                    float val = (
                        inputData[(y-2)*inputStep + x] +
                        inputData[(y-1)*inputStep + x] +
                        inputData[y*inputStep + x] +
                        inputData[(y+1)*inputStep + x] +
                        inputData[(y+2)*inputStep + x]
                    ) * 0.2f;
                    intermediateData[y*intermediateStep + x] = val;
                }
            }
        }
    }
}

// Horizontal blur
void horizontal_blur_optimized(const Mat &intermediate, Mat &output) {
    int width = intermediate.cols;
    int height = intermediate.rows;
    const float* interData = intermediate.ptr<float>(0);
    float* outData = output.ptr<float>(0);
    int interStep = (int)intermediate.step1();
    int outStep = (int)output.step1();

    __m256 w = _mm256_set1_ps(0.2f);

    #pragma omp parallel for schedule(dynamic)
    for (int yTile = 2; yTile < height - 2; yTile += TileHeight) {
        int yTileEnd = std::min(yTile + TileHeight, height - 2);
        for (int xTile = 2; xTile < width - 2; xTile += TileWidth) {
            int xTileEnd = std::min(xTile + TileWidth, width - 2);

            for (int y = yTile; y < yTileEnd; y++) {
                int x = xTile;
                for (; x <= xTileEnd - CHUNK; x += CHUNK) {
                    __m256 p0[CHUNK/VECTOR_WIDTH], p1[CHUNK/VECTOR_WIDTH], p2[CHUNK/VECTOR_WIDTH], p3[CHUNK/VECTOR_WIDTH], p4[CHUNK/VECTOR_WIDTH];

                    const float* base = interData + y*interStep + x;
                    for (int r = 0; r < CHUNK/VECTOR_WIDTH; r++) {
                        int offset = r*VECTOR_WIDTH;
                        p0[r] = _mm256_loadu_ps(base + offset - 2);
                        p1[r] = _mm256_loadu_ps(base + offset - 1);
                        p2[r] = _mm256_loadu_ps(base + offset);
                        p3[r] = _mm256_loadu_ps(base + offset + 1);
                        p4[r] = _mm256_loadu_ps(base + offset + 2);
                    }

                    for (int r = 0; r < CHUNK/VECTOR_WIDTH; r++) {
                        __m256 sum = _mm256_mul_ps(p0[r], w);
                        sum = _mm256_fmadd_ps(p1[r], w, sum);
                        sum = _mm256_fmadd_ps(p2[r], w, sum);
                        sum = _mm256_fmadd_ps(p3[r], w, sum);
                        sum = _mm256_fmadd_ps(p4[r], w, sum);
                        _mm256_storeu_ps(outData + y*outStep + x + r*VECTOR_WIDTH, sum);
                    }
                }

                // Handle remainder in chunks of 8
                for (; x <= xTileEnd - 8; x += 8) {
                    __m256 p0 = _mm256_loadu_ps(interData + y*interStep + x - 2);
                    __m256 p1 = _mm256_loadu_ps(interData + y*interStep + x - 1);
                    __m256 p2 = _mm256_loadu_ps(interData + y*interStep + x);
                    __m256 p3 = _mm256_loadu_ps(interData + y*interStep + x + 1);
                    __m256 p4 = _mm256_loadu_ps(interData + y*interStep + x + 2);

                    __m256 sum = _mm256_mul_ps(p0, w);
                    sum = _mm256_fmadd_ps(p1, w, sum);
                    sum = _mm256_fmadd_ps(p2, w, sum);
                    sum = _mm256_fmadd_ps(p3, w, sum);
                    sum = _mm256_fmadd_ps(p4, w, sum);

                    _mm256_storeu_ps(outData + y*outStep + x, sum);
                }

                // Scalar leftover
                for (; x < xTileEnd; x++) {
                    float val = (
                        interData[y*interStep + x - 2] +
                        interData[y*interStep + x - 1] +
                        interData[y*interStep + x] +
                        interData[y*interStep + x + 1] +
                        interData[y*interStep + x + 2]
                    ) * 0.2f;
                    outData[y*outStep + x] = val;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    // Single-threaded OpenCV for fair comparison
    setUseOptimized(false);
    setNumThreads(1);

    Mat img = imread("../input.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error: Could not read the image file!" << endl;
        return -1;
    }

    vector<int> sizes = {2048, 4096, 8192};
    vector<double> times_simd;
    vector<double> times_opencv;
    vector<double> gflops_per_cycle;

    for (int size : sizes) {
        Mat img_resized;
        resize(img, img_resized, Size(size, size));
        Mat img_float;
        img_resized.convertTo(img_float, CV_32F);

        Mat intermediate = Mat::zeros(img_float.size(), CV_32F);
        Mat output_simd = Mat::zeros(img_float.size(), CV_32F);
        Mat output_opencv;

        int64 t1 = getTickCount();
        vertical_blur_optimized(img_float, intermediate);
        horizontal_blur_optimized(intermediate, output_simd);
        double time_simd = (getTickCount() - t1) / getTickFrequency();

        int width = img_float.cols;
        int height = img_float.rows;
        double total_flops = (double)width * height * 20.0;

        // Compute GFLOPs per cycle directly
        // GFLOPs/s = (total_flops/time_simd)/1e9
        // GFLOPs/cycle = GFLOPs/s / CPU_FREQ_GHZ
        double gflops_per_cyc = ((total_flops/time_simd)/1e9) / CPU_FREQ_GHZ;

        times_simd.push_back(time_simd);
        gflops_per_cycle.push_back(gflops_per_cyc);

        t1 = getTickCount();
        blur(img_resized, output_opencv, Size(5,5));
        double time_opencv = (getTickCount() - t1) / getTickFrequency();
        times_opencv.push_back(time_opencv);

        cout << "Image size: " << size << " x " << size << endl;
        cout << "SIMD time: " << time_simd << " s" << endl;
        cout << "GFLOPs/cycle: " << gflops_per_cyc << endl;
        cout << "OpenCV time: " << time_opencv << " s" << endl;
        cout << "-------------------------------" << endl;

        Mat output_simd_8U;
        output_simd.convertTo(output_simd_8U, CV_8U);
        imwrite("../avg_kernel_output_images/output_simd_" + to_string(size) + ".jpg", output_simd_8U);
        imwrite("../avg_kernel_output_images/output_opencv_" + to_string(size) + ".jpg", output_opencv);
    }

    ofstream outfile("../average_performance_data.txt");
    outfile << "Size\tTime_SIMD(s)\tGFLOPs/cycle\tTime_OpenCV(s)\n";
    for (size_t i = 0; i < sizes.size(); ++i) {
        outfile << sizes[i] << "\t" << times_simd[i] << "\t" << gflops_per_cycle[i] << "\t" << times_opencv[i] << "\n";
    }
    outfile.close();

    return 0;
}

#include <opencv2/opencv.hpp>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include<fstream>

using namespace cv;
using namespace std;

// Adjust tile sizes based on cache size
const int TileHeight = 64;  // Adjust according to L1 cache size
const int TileWidth = 64;

void median_blur_simd(const Mat& input, Mat& output) {
    int width = input.cols;
    int height = input.rows;
    const float* inputData = input.ptr<float>();
    float* outputData = output.ptr<float>();
    int inputStep = input.step1();
    int outputStep = output.step1();

    // Parallelize outer loop using OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (int yTile = 2; yTile < height - 2; yTile += TileHeight) {
        int yTileEnd = min(yTile + TileHeight, height - 2);
        for (int xTile = 2; xTile < width - 2; xTile += TileWidth) {
            int xTileEnd = min(xTile + TileWidth, width - 2);

            for (int y = yTile; y < yTileEnd; y++) {
                for (int x = xTile; x < xTileEnd; x += 4) { // Process 4 pixels at a time
                    if (x + 3 >= xTileEnd) {
                        // Handle remaining pixels at the end of the tile
                        int remaining = xTileEnd - x;
                        for (int r = 0; r < remaining; ++r) {
                            // Process one pixel at position (x + r, y)
                            float buffer[25];
                            int idx = 0;
                            // Load 5x5 neighborhood
                            for (int ky = -2; ky <= 2; ky++) {
                                const float* rowPtr = inputData + (y + ky) * inputStep + x + r - 2;
                                for (int kx = 0; kx <= 4; kx++) {
                                    buffer[idx++] = rowPtr[kx];
                                }
                            }
                            // Use std::sort to sort the buffer
                            std::sort(buffer, buffer + 25);
                            // Get median
                            float median = buffer[12];
                            outputData[y * outputStep + x + r] = median;
                        }
                        break;
                    }

                    // Load 5x5 neighborhoods for four adjacent pixels into buffers
                    float buffer_a[25], buffer_b[25], buffer_c[25], buffer_d[25];

                    // Load data for the 5 rows
                    for (int ky = -2; ky <= 2; ky++) {
                        // Row pointers for four pixels
                        const float* rowPtr_a = inputData + (y + ky) * inputStep + x - 2;
                        const float* rowPtr_b = inputData + (y + ky) * inputStep + x + 2;
                        const float* rowPtr_c = inputData + (y + ky) * inputStep + x + 6;
                        const float* rowPtr_d = inputData + (y + ky) * inputStep + x + 10;

                        // Store first 5 elements from each row into buffers
                        buffer_a[(ky + 2) * 5 + 0] = rowPtr_a[0];
                        buffer_a[(ky + 2) * 5 + 1] = rowPtr_a[1];
                        buffer_a[(ky + 2) * 5 + 2] = rowPtr_a[2];
                        buffer_a[(ky + 2) * 5 + 3] = rowPtr_a[3];
                        buffer_a[(ky + 2) * 5 + 4] = rowPtr_a[4];

                        buffer_b[(ky + 2) * 5 + 0] = rowPtr_b[0];
                        buffer_b[(ky + 2) * 5 + 1] = rowPtr_b[1];
                        buffer_b[(ky + 2) * 5 + 2] = rowPtr_b[2];
                        buffer_b[(ky + 2) * 5 + 3] = rowPtr_b[3];
                        buffer_b[(ky + 2) * 5 + 4] = rowPtr_b[4];

                        buffer_c[(ky + 2) * 5 + 0] = rowPtr_c[0];
                        buffer_c[(ky + 2) * 5 + 1] = rowPtr_c[1];
                        buffer_c[(ky + 2) * 5 + 2] = rowPtr_c[2];
                        buffer_c[(ky + 2) * 5 + 3] = rowPtr_c[3];
                        buffer_c[(ky + 2) * 5 + 4] = rowPtr_c[4];

                        buffer_d[(ky + 2) * 5 + 0] = rowPtr_d[0];
                        buffer_d[(ky + 2) * 5 + 1] = rowPtr_d[1];
                        buffer_d[(ky + 2) * 5 + 2] = rowPtr_d[2];
                        buffer_d[(ky + 2) * 5 + 3] = rowPtr_d[3];
                        buffer_d[(ky + 2) * 5 + 4] = rowPtr_d[4];
                    }

                    // Use std::sort to sort the buffers
                    std::sort(buffer_a, buffer_a + 25);
                    std::sort(buffer_b, buffer_b + 25);
                    std::sort(buffer_c, buffer_c + 25);
                    std::sort(buffer_d, buffer_d + 25);

                    // Extract median
                    float median_a = buffer_a[12];
                    float median_b = buffer_b[12];
                    float median_c = buffer_c[12];
                    float median_d = buffer_d[12];

                    // Store median values in output
                    outputData[y * outputStep + x] = median_a;
                    outputData[y * outputStep + x + 1] = median_b;
                    outputData[y * outputStep + x + 2] = median_c;
                    outputData[y * outputStep + x + 3] = median_d;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    // Load input image in grayscale
    Mat img = imread("input.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error: Could not read the image file!" << endl;
        return -1;
    }

    // Define image sizes to test
    vector<int> sizes = {2048, 4096, 8192, 16384, 32768};

    // Initialize vectors to store performance data
    vector<double> times_simd;
    vector<double> times_opencv;

    for (int size : sizes) {
        // Resize image to current size
        Mat img_resized;
        resize(img, img_resized, Size(size, size));

        // Convert resized image to float32
        Mat img_float;
        img_resized.convertTo(img_float, CV_32F);

        // Initialize output images
        Mat output_simd = Mat::zeros(img_float.size(), CV_32F);
        Mat output_opencv;

        // Apply SIMD median filter implementation and measure the time
        int64 t1 = getTickCount();
        median_blur_simd(img_float, output_simd); // SIMD implementation
        double time_simd = (getTickCount() - t1) / getTickFrequency();

        // Store SIMD performance data
        times_simd.push_back(time_simd);

        // Apply OpenCV medianBlur implementation and measure the time
        t1 = getTickCount();
        medianBlur(img_resized, output_opencv, 5); // OpenCV implementation
        double time_opencv = (getTickCount() - t1) / getTickFrequency();

        // Store OpenCV performance data
        times_opencv.push_back(time_opencv);

        // Output performance data
        cout << "Image size: " << size << " x " << size << endl;
        cout << "SIMD implementation time: " << time_simd << " seconds" << endl;
        cout << "OpenCV implementation time: " << time_opencv << " seconds" << endl;
        cout << "-------------------------------" << endl;

        // Save SIMD output image for verification
        Mat output_simd_8U;
        output_simd.convertTo(output_simd_8U, CV_8U);
        string filename_simd = "../output_images/output_median_simd_" + to_string(size) + "x" + to_string(size) + ".jpg";
        imwrite(filename_simd, output_simd_8U);

        // Save OpenCV output image for verification
        string filename_opencv = "../output_images/output_median_opencv_" + to_string(size) + "x" + to_string(size) + ".jpg";
        imwrite(filename_opencv, output_opencv);
    }

    // Optionally, write performance data to a file for plotting
    ofstream outfile("../median_filter_performance_data.txt");
    outfile << "Size\tTime_SIMD(s)\tTime_OpenCV(s)\n";
    for (size_t i = 0; i < sizes.size(); ++i) {
        outfile << sizes[i] << "\t" << times_simd[i] << "\t" << times_opencv[i] << "\n";
    }
    outfile.close();

    return 0;
}
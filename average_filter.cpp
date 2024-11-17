#include <opencv2/opencv.hpp>
#include <immintrin.h>
#include <iostream>

using namespace cv;
using namespace std;

void apply_5x5_averaging_filter(const Mat &input, Mat &output) {
    __m256 reciprocal = _mm256_set1_ps(0.04f);  // 1/25 for normalization
    int width = input.cols;
    int height = input.rows;

    // Initialize output image
    output = Mat::zeros(input.size(), CV_32F);

    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x += 8) {  // Process 8 pixels at a time to align with AVX

            // Initialize the sum register to zero for each of the 8 pixels
            __m256 sum = _mm256_setzero_ps();

            // Unroll the loop to process the 5x5 neighborhood
            // Row -2
            __m256 row1a = _mm256_loadu_ps(input.ptr<float>(y - 2) + x - 2);
            __m256 row1b = _mm256_loadu_ps(input.ptr<float>(y - 2) + x - 1);
            __m256 row1c = _mm256_loadu_ps(input.ptr<float>(y - 2) + x);
            __m256 row1d = _mm256_loadu_ps(input.ptr<float>(y - 2) + x + 1);
            __m256 row1e = _mm256_loadu_ps(input.ptr<float>(y - 2) + x + 2);
            sum = _mm256_add_ps(sum, row1a);
            sum = _mm256_add_ps(sum, row1b);
            sum = _mm256_add_ps(sum, row1c);
            sum = _mm256_add_ps(sum, row1d);
            sum = _mm256_add_ps(sum, row1e);

            // Row -1
            __m256 row2a = _mm256_loadu_ps(input.ptr<float>(y - 1) + x - 2);
            __m256 row2b = _mm256_loadu_ps(input.ptr<float>(y - 1) + x - 1);
            __m256 row2c = _mm256_loadu_ps(input.ptr<float>(y - 1) + x);
            __m256 row2d = _mm256_loadu_ps(input.ptr<float>(y - 1) + x + 1);
            __m256 row2e = _mm256_loadu_ps(input.ptr<float>(y - 1) + x + 2);
            sum = _mm256_add_ps(sum, row2a);
            sum = _mm256_add_ps(sum, row2b);
            sum = _mm256_add_ps(sum, row2c);
            sum = _mm256_add_ps(sum, row2d);
            sum = _mm256_add_ps(sum, row2e);

            // Row 0
            __m256 row3a = _mm256_loadu_ps(input.ptr<float>(y) + x - 2);
            __m256 row3b = _mm256_loadu_ps(input.ptr<float>(y) + x - 1);
            __m256 row3c = _mm256_loadu_ps(input.ptr<float>(y) + x);
            __m256 row3d = _mm256_loadu_ps(input.ptr<float>(y) + x + 1);
            __m256 row3e = _mm256_loadu_ps(input.ptr<float>(y) + x + 2);
            sum = _mm256_add_ps(sum, row3a);
            sum = _mm256_add_ps(sum, row3b);
            sum = _mm256_add_ps(sum, row3c);
            sum = _mm256_add_ps(sum, row3d);
            sum = _mm256_add_ps(sum, row3e);

            // Row +1
            __m256 row4a = _mm256_loadu_ps(input.ptr<float>(y + 1) + x - 2);
            __m256 row4b = _mm256_loadu_ps(input.ptr<float>(y + 1) + x - 1);
            __m256 row4c = _mm256_loadu_ps(input.ptr<float>(y + 1) + x);
            __m256 row4d = _mm256_loadu_ps(input.ptr<float>(y + 1) + x + 1);
            __m256 row4e = _mm256_loadu_ps(input.ptr<float>(y + 1) + x + 2);
            sum = _mm256_add_ps(sum, row4a);
            sum = _mm256_add_ps(sum, row4b);
            sum = _mm256_add_ps(sum, row4c);
            sum = _mm256_add_ps(sum, row4d);
            sum = _mm256_add_ps(sum, row4e);

            // Row +2
            __m256 row5a = _mm256_loadu_ps(input.ptr<float>(y + 2) + x - 2);
            __m256 row5b = _mm256_loadu_ps(input.ptr<float>(y + 2) + x - 1);
            __m256 row5c = _mm256_loadu_ps(input.ptr<float>(y + 2) + x);
            __m256 row5d = _mm256_loadu_ps(input.ptr<float>(y + 2) + x + 1);
            __m256 row5e = _mm256_loadu_ps(input.ptr<float>(y + 2) + x + 2);
            sum = _mm256_add_ps(sum, row5a);
            sum = _mm256_add_ps(sum, row5b);
            sum = _mm256_add_ps(sum, row5c);
            sum = _mm256_add_ps(sum, row5d);
            sum = _mm256_add_ps(sum, row5e);

            // Divide by 25 to get the average
            sum = _mm256_mul_ps(sum, reciprocal);

            // Store the result in the output image
            _mm256_storeu_ps(output.ptr<float>(y) + x, sum);
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

    // Convert image to float32 for SIMD processing
    Mat img_float;
    img.convertTo(img_float, CV_32F);

    // Output images for both implementations
    Mat output_simd, output_opencv;

    // Apply SIMD implementation and measure the time
    int64 t1 = getTickCount();
    apply_5x5_averaging_filter(img_float, output_simd);
    double time_simd = (getTickCount() - t1) / getTickFrequency();
    cout << "SIMD implementation time: " << time_simd << " seconds" << endl;

    // Convert SIMD output back to 8-bit for comparison and saving
    output_simd.convertTo(output_simd, CV_8U);
    imwrite("../output_simd.jpg", output_simd);

    // Apply OpenCV implementation and measure the time
    t1 = getTickCount();
    blur(img, output_opencv, Size(5, 5));
    double time_opencv = (getTickCount() - t1) / getTickFrequency();
    cout << "OpenCV implementation time: " << time_opencv << " seconds" << endl;

    // Save OpenCV output
    imwrite("../output_opencv.jpg", output_opencv);

    // Display comparison results
    cout << "Performance comparison:" << endl;
    cout << "SIMD implementation time: " << time_simd << " seconds" << endl;
    cout << "OpenCV implementation time: " << time_opencv << " seconds" << endl;

    return 0;
}

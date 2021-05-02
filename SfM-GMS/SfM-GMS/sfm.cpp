// Program0.cpp

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;

// main - a quick test of OpenCV			
int main(int argc, char* argv[])
{
    Mat image = imread("../SourceImages/L1.jpg");

    namedWindow("Original Image");
    imshow("Original Image", image);
    waitKey(0);

    for (int r = 0; r < image.rows; r++) {
        for (int c = 0; c < image.cols; c++) {
            for (int b = 0; b < 3; b++) {
                image.at<Vec3b>(r, c)[b] = 255 - image.at<Vec3b>(r, c)[b];
            }
        }
    }

    namedWindow("Photonegative");
    imshow("Photonegative", image);
    waitKey(0);

    return 0;
}
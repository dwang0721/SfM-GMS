#pragma once

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

#include "FeatureMatchUtil.h"
#include "CalibrationUtil.h"
#include "SfMUtil.h"

using namespace cv;
using namespace std;

// ----------------------------------
// ----------- Constant -------------
// ----------------------------------
const String parserKeys =
"{help h usage ? |      | print this message   }"
"{@image1        |      | image1 for compare   }"
"{@image2        |<none>| image2 for compare   }"
"{algorithm      |      | LOGOS or GMS         }"
;

enum { DEFAULT_SIFT = 1, GMS = 2, LOGOS = 3};

vector<string> files = {"../CalibrationImages/IMG_0.jpg", 
                        "../CalibrationImages/IMG_1.jpg",
                        "../CalibrationImages/IMG_2.jpg",
                        "../CalibrationImages/IMG_3.jpg",
                        "../CalibrationImages/IMG_4.jpg",
                        "../CalibrationImages/IMG_5.jpg",
                        "../CalibrationImages/IMG_6.jpg",
                        "../CalibrationImages/IMG_7.jpg",
                        "../CalibrationImages/IMG_8.jpg",
                        "../CalibrationImages/IMG_9.jpg",
                        };


vector<Mat> rvecs, tvecs;
Size board_size(6,9);
vector<KeyPoint> keypoints1, keypoints2;
Mat descriptors1, descriptors2;

// ----------------------------------
// ---- Function Declaration --------
// ----------------------------------
void processParser(CommandLineParser parser);
void printHelp();
void blurDistant(Mat& img1, Mat& img2, Mat& disparity);
Mat img_rotate(Mat src, double angle);

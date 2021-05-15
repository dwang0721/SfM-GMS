#pragma once

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "FeatureMatchUtil.h"
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

enum { STEREO_SGBM = 1, LOGOS = 2, GMS = 3};

// ----------------------------------
// ---- Function Declaration --------
// ----------------------------------
void processParser(CommandLineParser parser);
void printHelp();
void stereo_match(const Mat &img1, const Mat &img2, Mat &disparity);
void blurDistant(Mat& img1, Mat& img2, Mat& disparity);

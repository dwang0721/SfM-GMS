#pragma once
#pragma once

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types_c.h>
#include <iostream>
#include <conio.h>
#include <opencv2/photo/cuda.hpp>

#include <iostream>

#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

// ----------------------------------
// ----------- Constant -------------
// ----------------------------------

// ----------------------------------
// ---- Function Declaration --------
// ----------------------------------
void processParser(CommandLineParser parser);
void printHelp();
void stereo_match(const Mat& img1, const Mat& img2, Mat& disparity);
void blurDistant(Mat& img1, Mat& img2, Mat& disparity);
int runDisparityMap();

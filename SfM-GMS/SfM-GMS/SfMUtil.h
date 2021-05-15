#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// ----------------------------------
// ----------- Constant -------------
// ----------------------------------


// ----------------------------------
// ---- Function Declaration --------
// ----------------------------------
void structureFromMotion(Mat& img1, Mat& img2);
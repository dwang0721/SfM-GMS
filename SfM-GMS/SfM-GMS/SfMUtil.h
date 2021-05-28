#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace viz;

// ----------------------------------
// ----------- Constant -------------
// ----------------------------------


// ----------------------------------
// ---- Function Declaration --------
// ----------------------------------
void structureFromMotion(Mat& img1, Mat& img2, Mat& cameraMatrix, Mat& distCoeffs, vector<Vec3d>& points3D, bool drawMatchResult, int algo_enum);
Mat computeProjMat(Mat cameraMatrix, Mat rotationMatrix, Mat transMatrix);
Vec3d solveTriangulation(const cv::Mat& p1, const cv::Mat& p2, const cv::Vec2d& u1, const cv::Vec2d& u2);
void triangulate_SVD(const cv::Mat& p1, const cv::Mat& p2, const std::vector<cv::Vec2d>& pts1, const std::vector<cv::Vec2d>& pts2, std::vector<cv::Vec3d>& pts3D);
void triangulate_OpenCV(const cv::Mat& projMatrix1, const cv::Mat& projMatrix2, const std::vector<cv::Vec2d>& undistCoords1, const std::vector<cv::Vec2d>& undistCoords2, std::vector<cv::Vec3d>& pts3D);
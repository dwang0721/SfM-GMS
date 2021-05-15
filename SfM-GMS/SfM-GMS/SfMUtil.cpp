#include "SfMUtil.h"
#include "FeatureMatchUtil.h"

void structureFromMotion(Mat& img1, Mat& img2)
{
    // key points
    std::vector<cv::KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    vector<DMatch> matches;

    // detect key features using SIFT
    cout << "Start detecting..." << endl;
    SIFTDetectAndCompute(img1, kpts1, desc1);
    SIFTDetectAndCompute(img2, kpts2, desc2);
    cout << "Done detecting" << endl;

    // match descriptors
    cout << "Start matching..." << endl;
    bruteForceMatch(desc1, desc2, matches);
    cout << "Done matching" << endl;

    // draw the result
    Mat matchImg;
    cout << "Drawing matching result..." << endl;
    drawMatches(img1, kpts1, img2, kpts2, matches, matchImg);
    namedWindow("result", WINDOW_NORMAL);
    resizeWindow("result", matchImg.cols / 8, matchImg.rows / 8);
    imshow("result", matchImg);
    waitKey(0);

    //uv coordinates
    cout << "Drawing matching result..." << endl;
    std::vector<cv::Point2f> coords1, coords2;
    for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
        // Get the position of left key points
        float x = kpts1[it->queryIdx].pt.x;
        float y = kpts1[it->queryIdx].pt.y;
        coords1.push_back(cv::Point2f(x, y));
        // Get the position of right key points
        x = kpts2[it->trainIdx].pt.x;
        y = kpts2[it->trainIdx].pt.y;
        coords2.push_back(cv::Point2f(x, y));
    }

    // find camera matrix
    Mat cameraMatrix;

    // find essential matrix
    Mat inliers;
    Mat essential = findEssentialMat(coords1, coords2, cameraMatrix, cv::RANSAC, 0.9, 1.0, inliers);
}
#include "SfMUtil.h"
#include "FeatureMatchUtil.h"

void structureFromMotion(Mat& img1, Mat& img2, Mat& cameraMatrix)
{
    // key points
    std::vector<cv::KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    vector<DMatch> matches;

    // detect key features using SIFT
    cout << "Start SIFT detecting..." << endl;
    SIFTDetectAndCompute(img1, kpts1, desc1);
    SIFTDetectAndCompute(img2, kpts2, desc2);
    cout << "Done SIFT detecting" << endl;

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

    // find essential matrix
    Mat inliers;
    Mat essential = findEssentialMat(coords1, coords2, cameraMatrix, cv::RANSAC, 0.9, 1.0, inliers);
    cout<< "essentail:\n" << essential <<endl;

    // recover rotation and translation from essential matrix
    Mat rotation, translation;
    recoverPose(essential, coords1, coords2, cameraMatrix, rotation, translation, inliers);
    cout << "rotation:\n" << rotation << endl;
    cout << "translation:\n" << translation << endl;

    // calculate projection matrix
    cout << "Calculating Projection Matrix..." << endl;
    Mat projMatrix1 (3, 4, CV_64F, 0.);
    cv::Mat diag(cv::Mat::eye(3, 3, CV_64F));
    diag.copyTo(projMatrix1(cv::Rect(0, 0, 3, 3)));
    Mat projMatrix2 = computeProjMat(cameraMatrix, rotation, translation);
    cout << "Projection 1: \n" << projMatrix1 << endl;
    cout << "Projection 2: \n" << projMatrix2 << endl;
    waitKey(0);
}


Mat computeProjMat(Mat cameraMatrix, Mat rotationMatrix, Mat transMatrix)
{
    Mat RTMat(3, 4, CV_64F);
    hconcat(rotationMatrix, transMatrix, RTMat);
    return (cameraMatrix * RTMat);
}
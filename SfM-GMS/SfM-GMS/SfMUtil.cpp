#include "SfMUtil.h"
#include "FeatureMatchUtil.h"

void structureFromMotion(Mat& img1, Mat& img2, Mat& cameraMatrix, Mat& distCoeffs, vector<Vec3d>& points3D, bool drawMatchResult, int algo_enum)
{
    // key points
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    vector<DMatch> matches;

    // detect key features using SIFT
    if (algo_enum == 1){
        SIFT_matchBF(img1, img2, kpts1, kpts2, desc1, desc2, matches, drawMatchResult);
    }

    if (algo_enum == 2){
        SIFT_matchGMS(img1, img2, kpts1, kpts2, desc1, desc2, matches, drawMatchResult);
    }

    if (algo_enum == 3) {
        SIFT_matchLOGOS(img1, img2, kpts1, kpts2, desc1, desc2, matches, drawMatchResult);
    }

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

    // find essential matrix using RANSAC
    Mat inliers;
    Mat essential = findEssentialMat(coords1, coords2, cameraMatrix, cv::RANSAC, 0.7, 1.0, inliers);
    cout<< "\nEssentail:\n" << essential <<endl;

    // Recovers the relative camera rotation and the translation from an estimated essential matrix and the corresponding points in two images, 
    // using cheirality check. Returns the number of inliers that pass the check.
    Mat rotation, translation;
    recoverPose(essential, coords1, coords2, cameraMatrix, rotation, translation, inliers);
    cout << "\nRotation:\n" << rotation << endl;
    cout << "\nTranslation:\n" << translation << endl;

    // calculate projection matrix
    cout << "Calculating Projection Matrix..." << endl;

    //-------Project Matrix without intrinsic-----------
    cv::Mat projMatrix1(3, 4, CV_64F, 0.); 
    cv::Mat diag(cv::Mat::eye(3, 3, CV_64F));
    diag.copyTo(projMatrix1(cv::Rect(0, 0, 3, 3)));

    cv::Mat projMatrix2(3, 4, CV_64F); 
    rotation.copyTo(projMatrix2(cv::Rect(0, 0, 3, 3)));
    translation.copyTo(projMatrix2.colRange(3, 4));
    //-----------------
    
    cout << "\nProjectionMatrix 1: \n" << projMatrix1 << endl;
    cout << "\nProjectionMatrix 2: \n" << projMatrix2 << endl; 

    // take inliers 
    cout << "\nUn distort inliers..." << endl;
    std::vector<cv::Vec2d> inlierPts1;
    std::vector<cv::Vec2d> inlierPts2;
    for (int i = 0; i < inliers.rows; i++) {
        if (inliers.at<uchar>(i)) {
            inlierPts1.push_back(cv::Vec2d(coords1[i].x, coords1[i].y));
            inlierPts2.push_back(cv::Vec2d(coords2[i].x, coords2[i].y));
        }
    }

    // un-distortion
    std::vector<cv::Vec2d> undistCoords1, undistCoords2;
    cv::undistortPoints(inlierPts1, undistCoords1, cameraMatrix, distCoeffs);
    cv::undistortPoints(inlierPts2, undistCoords2, cameraMatrix, distCoeffs);

    // triangulation    
    triangulate_OpenCV(projMatrix1, projMatrix2, undistCoords1, undistCoords2, points3D);
}


Mat computeProjMat(Mat cameraMatrix, Mat rotationMatrix, Mat transMatrix)
{
    Mat RTMat(3, 4, CV_64F);
    hconcat(rotationMatrix, transMatrix, RTMat);
    return (cameraMatrix * RTMat);
}

Vec3d solveTriangulation(const cv::Mat& p1, const cv::Mat& p2, const cv::Vec2d& u1, const cv::Vec2d& u2) {
    // system of equations assuming image=[u,v] and X=[x,y,z,1]
    // from u(p3.X)= p1.X and v(p3.X)=p2.X
    cv::Matx43d A(u1(0) * p1.at<double>(2, 0) - p1.at<double>(0, 0),
        u1(0) * p1.at<double>(2, 1) - p1.at<double>(0, 1),
        u1(0) * p1.at<double>(2, 2) - p1.at<double>(0, 2),
        u1(1) * p1.at<double>(2, 0) - p1.at<double>(1, 0),
        u1(1) * p1.at<double>(2, 1) - p1.at<double>(1, 1),
        u1(1) * p1.at<double>(2, 2) - p1.at<double>(1, 2),
        u2(0) * p2.at<double>(2, 0) - p2.at<double>(0, 0),
        u2(0) * p2.at<double>(2, 1) - p2.at<double>(0, 1),
        u2(0) * p2.at<double>(2, 2) - p2.at<double>(0, 2),
        u2(1) * p2.at<double>(2, 0) - p2.at<double>(1, 0),
        u2(1) * p2.at<double>(2, 1) - p2.at<double>(1, 1),
        u2(1) * p2.at<double>(2, 2) - p2.at<double>(1, 2));

    cv::Matx41d B(p1.at<double>(0, 3) - u1(0) * p1.at<double>(2, 3),
        p1.at<double>(1, 3) - u1(1) * p1.at<double>(2, 3),
        p2.at<double>(0, 3) - u2(0) * p2.at<double>(2, 3),
        p2.at<double>(1, 3) - u2(1) * p2.at<double>(2, 3));

    // X contains the 3D coordinate of the reconstructed point
    cv::Vec3d X;
    // solve AX=B
    cv::solve(A, B, X, cv::DECOMP_SVD);
    return X;
}

void triangulate_SVD(const cv::Mat& p1, const cv::Mat& p2, const std::vector<cv::Vec2d>& pts1, const std::vector<cv::Vec2d>& pts2, std::vector<cv::Vec3d>& pts3D) {
    cout << "Triangulation..." << endl;
    for (int i = 0; i < pts1.size(); i++) {
        pts3D.push_back(solveTriangulation(p1, p2, pts1[i], pts2[i]));
    }
}

void triangulate_OpenCV(const cv::Mat& projMatrix1, const cv::Mat& projMatrix2, const std::vector<cv::Vec2d>& undistCoords1, const std::vector<cv::Vec2d>& undistCoords2, std::vector<cv::Vec3d>& pts3D) {
    cout << "\nTriangulation..." << endl;
    cv::Mat triangCoords4D;
    cv::triangulatePoints(projMatrix1, projMatrix2, undistCoords1, undistCoords2, triangCoords4D);

    // recover 3d coordinates
    for (int i = 0; i < triangCoords4D.cols; i++) {
        Vec4d point4D = triangCoords4D.col(i);
        double x = point4D[0] / point4D[3];
        double y = point4D[1] / point4D[3];
        double z = point4D[2] / point4D[3];
        Vec3d p(x, y, z);
        pts3D.push_back(p);
        cout << "point 3d" << i << ": " << p << endl;
    }
    cout << "Number of points found: " << pts3D.size() << endl;
}
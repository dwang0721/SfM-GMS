#include "main.h"
int alg = DEFAULT_SIFT;

// Tasks:
// 1. parser and image input 
// 2. LOGOS application for SfM <-Purrnima
// 3. GMS application for SfM  <-- Neil
// 4. baseline implementation
//     a. how to use match points for disparity map <--- Ahmed
//     b. how to use match points for reconstruction 3d. <--- Di

			
int main(int argc, char* argv[])
{
    // ToDo: parser implementation need
    // CommandLineParser parser(argc, argv, parserKeys);
    // processParser(parser);
    
    Mat img1 = imread("../SourceImages/Disparity_L.jpg");
    Mat img2 = imread("../SourceImages/Disparity_R.jpg");
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;

    // ------------- Feature Match: Logos and GMS ------------------ 
    cout << "-------- Feature Match: GMS vs LOGOS ---------" << endl; 
    cout << img2.cols << endl;
    cout << img2.rows << endl;
    // GMS match
    cout << "-------- Normal Camera Change ---------" << endl;
    vector<DMatch> matches, matchesGMS;
    SIFT_matchBF(img1, img2, kpts1, kpts2, desc1, desc2, matches, true);
    SIFT_matchGMS(img1, img2,  kpts1, kpts2, desc1, desc2, matchesGMS, true);

    // match with rotation
    cout << "-------- With Rotation ---------" << endl;
    Mat rot_img2 = img_rotate(img2, 180);
    vector<DMatch> matches2, matchesGMS2;
    SIFT_matchBF(img1, rot_img2, kpts1, kpts2, desc1, desc2, matches, true);
    SIFT_matchGMS(img1, rot_img2, kpts1, kpts2, desc1, desc2, matchesGMS, true);

    // match with scale
    cout << "-------- Change Scale ---------" << endl;
    Mat scale_img2;
    resize(img2,scale_img2, Size(1000,1000));
    vector<DMatch> matches3, matchesGMS3;
    SIFT_matchBF(img1, scale_img2, kpts1, kpts2, desc1, desc2, matches3, true);
    SIFT_matchGMS(img1, scale_img2, kpts1, kpts2, desc1, desc2, matchesGMS3, true);

    // Logos Match
    vector<DMatch> matchesLOGOS;
    SIFT_matchLOGOS(img1, img2, kpts1, kpts2, desc1, desc2, matchesLOGOS, true);

    // ----------- calibration -----------------
    cout << "\n----------- Structure From Motion ----------" << endl;
    Mat cameraMatrix, distCoeffs;
    vector<vector<Point3f>> objectPoints;
    vector<vector<Point2f>> imagePoints;

    addChessboardPoints(files, board_size, objectPoints, imagePoints, false);
    Mat img = cv::imread("../CalibrationImages/IMG_0.jpg");
    calibrateCamera(objectPoints, // the 3D points
                    imagePoints,  // the image points
                    img.size(),   // image size
                    cameraMatrix, // output camera matrix
                    distCoeffs,   // output distortion matrix
                    rvecs, tvecs // Rs, Ts 
                    );
    cout << "\nCamera Matrix :\n" << cameraMatrix << endl;

    // --------- SfM Implementation --------------
    Mat imgL = imread("../SourceImages/PikaBun1.jpg");
    Mat imgR = imread("../SourceImages/PikaBun4.jpg");
    vector<Vec3d> points3D;
    structureFromMotion(imgL, imgR, cameraMatrix, distCoeffs, points3D, true, LOGOS);
    //structureFromMotion(imgL, imgR, cameraMatrix, distCoeffs, points3D, true, GMS);
    //structureFromMotion(imgL, imgR, cameraMatrix, distCoeffs, points3D, true, DEFAULT_SIFT); // <-- change the algorithm here

    // --------- Draw 3d point cloud --------------
    Viz3d window;
    window.showWidget("coordinate", viz::WCoordinateSystem());
    window.setBackgroundColor(cv::viz::Color::black());
    window.showWidget("points", viz::WCloud(points3D, viz::Color::white()));
    window.spin();
    waitKey(0);

    // -------------- Ahmed: disparity map -------------------
    runDisparityMap();
    return 0;
}

// ----------------------------------
// ---- Function Implementation -----
// ----------------------------------
void processParser(CommandLineParser parser){
    if (parser.has("help"))
    {
        printHelp();
    }

    if (parser.has("algorithm"))
    {
        std::string _alg = parser.get<string>("algorithm");
        if (_alg == "SIFT"){
            alg = DEFAULT_SIFT;
        }
        // continue other algorithm
    }
}

void printHelp(){
   printf("\nhelp message here!");
}

Mat img_rotate(Mat src, double angle) {
    Mat dst;      //Mat object for output image file
    Point2f pt(src.cols / 2., src.rows / 2.);          //point from where to rotate    
    Mat r = getRotationMatrix2D(pt, angle, 1.0);      //Mat object for storing after rotation
    warpAffine(src, dst, r, Size(src.cols, src.rows));  ///applie an affine transforation to image.
    return dst;         //returning Mat object for output image file
}

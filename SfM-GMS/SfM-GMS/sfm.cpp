#include "sfm.h"
int alg = STEREO_SGBM;

// Tasks:
// 1. parser and image input 
// 2. LOGOS application for SfM <-Purrnima
// 3. GMS application for SfM  <-- Neil
// 4. baseline implementation
//     a. how to use match points for disparity map <--- Ahmed
//     b. how to use match points for reconstruction 3d. <--- Di

// main - a quick test of OpenCV			
int main(int argc, char* argv[])
{
    // ToDo: parser implementation need
    // CommandLineParser parser(argc, argv, parserKeys);
    // processParser(parser);

    // ------------- Feature Match: Logos and GMS-------------------  
    Mat img1= imread("../SourceImages/Disparity_L.jpg");
    Mat img2 = imread("../SourceImages/Disparity_R.jpg");

    // GMS match
    vector<DMatch> matchesGMS,matches;
    SIFT_match(img1, img2, matches, true);
    SIFT_matchGMA(img1, img2, matchesGMS, true, false, false);
    Mat rot_img2 = img_rotate(img2,180.0);
    vector<DMatch> matchesLOGOS2,matchesGMS2,matches2;
    SIFT_match(img1, rot_img2, matches2, true);
    SIFT_matchGMA(img1, rot_img2, matchesGMS2, true, true, true);
    SIFT_matchLOGOS(img1, rot_img2, matchesLOGOS2, true);


    // Logos Match
    /*vector<DMatch> matchesLOGOS;
    SIFT_matchLOGOS(img1, img2, matchesLOGOS, true);

    // ----------- calibration -----------------  
    Mat cameraMatrix, distCoeffs;
    vector<vector<Point3f>> objectPoints;
    vector<vector<Point2f>> imagePoints;

    addChessboardPoints(files, board_size, objectPoints, imagePoints);
    Mat img = cv::imread("../CalibrationImages/IMG_0.jpg");
    calibrateCamera(objectPoints, // the 3D points
                    imagePoints,  // the image points
                    img.size(),   // image size
                    cameraMatrix, // output camera matrix
                    distCoeffs,   // output distortion matrix
                    rvecs, tvecs // Rs, Ts 
                    );
    cout << "Camera Matrix :\n" << cameraMatrix << endl;
    waitKey(0);

    // --------- SfM Implementation --------------
    Mat imgL = imread("../SourceImages/PikaBun1.jpg");
    Mat imgR = imread("../SourceImages/PikaBun4.jpg");
    vector<Vec3d> points3D;
    structureFromMotion(imgL, imgR, cameraMatrix, distCoeffs, points3D);

    // draw points on screen
    Viz3d window;
    window.showWidget("coordinate", viz::WCoordinateSystem());
    window.setBackgroundColor(cv::viz::Color::black());
    window.showWidget("points", viz::WCloud(points3D, viz::Color::green()));
    window.spin();
    waitKey(0);*/

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
        if (_alg == "SGBM"){
            alg = STEREO_SGBM;
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


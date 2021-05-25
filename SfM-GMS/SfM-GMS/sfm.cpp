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
    vector<DMatch> matchesGMS;
    SIFT_matchGMA(img1, img2, matchesGMS, true);

    // Logos Match
    vector<DMatch> matchesLOGOS;
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
    waitKey(0);

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


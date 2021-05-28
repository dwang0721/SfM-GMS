#include "sfm.h"
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
    vector<DMatch> matches;

    // ------------- Feature Match: Logos and GMS ------------------ 
    cout << "-------- Feature Match: GMS vs LOGOS ---------" << endl; 

    // GMS match
    vector<DMatch> matchesGMS;
    SIFT_matchGMS(img1, img2,  kpts1, kpts2, desc1, desc2, matchesGMS, true);

    // Logos Match
    vector<DMatch> matchesLOGOS;
    SIFT_matchLOGOS(img1, img2, kpts1, kpts2, desc1, desc2, matchesLOGOS, true);

    // -------------- Ahmed: disparity map -------------------

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
    structureFromMotion(imgL, imgR, cameraMatrix, distCoeffs, points3D, true, LOGOS); // <-- change the algorithm here

    // --------- Draw 3d point cloud --------------
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
        if (_alg == "SIFT"){
            alg = DEFAULT_SIFT;
        }
        // continue other algorithm
    }
}

void printHelp(){
   printf("\nhelp message here!");
}

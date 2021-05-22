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
    //CommandLineParser parser(argc, argv, parserKeys);
    //processParser(parser);


    // ------------- Feature Match: Logos and GMS-------------------  
    const Mat img1 = imread("../SourceImages/Disparity_L.jpg");
    const Mat img2 = imread("../SourceImages/Disparity_R.jpg");

    //Ptr<SIFT> detector = SIFT::create();
    //vector<KeyPoint> keypoints1, keypoints2;
    //Mat descriptor1, descriptor2;
    //// Obtain keypoints and descriptors
    //detector->detectAndCompute(img1, noArray(), keypoints1, descriptor1);
    //detector->detectAndCompute(img2, noArray(), keypoints2, descriptor2);
    //Ptr<BFMatcher> matcher1 = BFMatcher::create();
    //vector<DMatch> matches, matchesGMS;
    //// Match two images' descriptors
    //matcher1->match(descriptor1, descriptor2, matches);
    //// GMS
    //cv::xfeatures2d::matchGMS(img1.size(), img2.size(), keypoints1, keypoints2, matches, matchesGMS);

    //Ptr<FlannBasedMatcher> matcher2 = FlannBasedMatcher::create();
    //BOWKMeansTrainer bow(50);
    //Mat dict = bow.cluster(descriptor1);
    //vector<int> nn1, nn2;
    //vector<DMatch> m1, m2, logosMatches;
    //matcher2->add(dict);
    //matcher2->match(descriptor1, m1);
    //matcher2->match(descriptor2, m2);

    //for (auto m : m1) {
    //    nn1.push_back(m.trainIdx);
    //}
    //for (auto m : m2) {
    //    nn2.push_back(m.trainIdx);
    //}
    //// LOGOS
    //cv::xfeatures2d::matchLOGOS(keypoints1, keypoints2, nn1, nn2, logosMatches);

    //Mat image_show1;
    //drawMatches(img1, keypoints1, img2, keypoints2, matchesGMS, image_show1);
    //namedWindow("GMS", WINDOW_NORMAL);
    //
    //// Scale down the window size
    //resizeWindow("GMS", image_show1.cols / SCALE, image_show1.rows / SCALE);
    //imshow("GMS", image_show1);
    //waitKey(0);

    //Mat image_show2;
    //drawMatches(img1, keypoints1, img2, keypoints2, logosMatches, image_show2, Scalar::all(-1),
    //    Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //namedWindow("LOGOS", WINDOW_NORMAL);
    //resizeWindow("LOGOS", image_show2.cols / SCALE, image_show2.rows / SCALE);  // SCALE = 32 worked well for my system
    //imshow("LOGOS", image_show2);
    //waitKey(0);


    // ----------- calibration -----------------    
    addChessboardPoints(files, board_size, objectPoints, imagePoints);
    Mat img = cv::imread("../CalibrationImages/IMG_0.jpg");
    calibrateCamera(objectPoints, // the 3D points
                    imagePoints,  // the image points
                    img.size(),   // image size
                    cameraMatrix, // output camera matrix
                    distCoeffs,   // output distortion matrix
                    rvecs, tvecs // Rs, Ts 
                    );

    // --------- SfM Implementation --------------
    Mat imgL = imread("../SourceImages/pikaL.jpg");
    Mat imgR = imread("../SourceImages/pikaR.jpg");
    structureFromMotion(imgL, imgR, cameraMatrix);

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


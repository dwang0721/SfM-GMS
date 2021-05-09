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
    CommandLineParser parser(argc, argv, parserKeys);
    processParser(parser);

    // initialize input
    const Mat img1 = imread("../SourceImages/Disparity_L.jpg");
    const Mat img2 = imread("../SourceImages/Disparity_R.jpg");
    Mat disp;

    namedWindow("Left Eye Image", WINDOW_NORMAL);
    resizeWindow("Left Eye Image", img1.cols/4, img1.rows/4);
    imshow("Left Eye Image", img1);
    
    namedWindow("Right Eye Image", WINDOW_NORMAL);
    resizeWindow("Right Eye Image", img2.cols/4, img2.rows/4);
    imshow("Right Eye Image", img2);    
    waitKey(0);

    // Stereo: disparity map
    cout << "Start stereo matching..." << endl;
    stereo_match(img1, img2, disp);
    namedWindow("Disparity Map", WINDOW_NORMAL);
    imshow("Disparity Map", disp);
    cout << "Done matching" << endl;
    waitKey(0);

    // ToDo: SfM

    // ToDo: Disparity Map

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

void stereo_match(const Mat &img1, const Mat &img2, Mat &disparity){
    // place holder algorithm
    Mat g1, g2;
    cvtColor(img1, g1, COLOR_BGR2GRAY);
    cvtColor(img2, g2, COLOR_BGR2GRAY);

    Ptr<StereoBM> sbm = StereoBM::create(16, 9);
    sbm->setNumDisparities(112);
    sbm->setPreFilterSize(5);
    sbm->setPreFilterCap(61);
    sbm->setMinDisparity(-39);
    sbm->setTextureThreshold(507);
    sbm->setUniquenessRatio(0);
    sbm->setSpeckleWindowSize(0);
    sbm->setSpeckleRange(8);
    sbm->setDisp12MaxDiff(1);
    sbm->compute(g1, g2, disparity);
    normalize(disparity, disparity, 0, 255, NORM_MINMAX, CV_8U);
}
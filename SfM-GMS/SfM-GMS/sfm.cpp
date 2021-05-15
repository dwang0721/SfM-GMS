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
    //const Mat img1 = imread("../SourceImages/Disparity_L.jpg");
    //const Mat img2 = imread("../SourceImages/Disparity_R.jpg");
    //Mat disp;

    //Mat img3 = imread("../SourceImages/view0-1.png");
    //Mat img4 = imread("../SourceImages/view2-1.png");
    //cout << "Start background blurring..." << endl;
    //blurDistant(img3, img4, disp);
    //cout << "Done" << endl;
    //waitKey(0);

    //namedWindow("Left Eye Image", WINDOW_NORMAL);
    //resizeWindow("Left Eye Image", img1.cols/4, img1.rows/4);
    //imshow("Left Eye Image", img1);
    //
    //namedWindow("Right Eye Image", WINDOW_NORMAL);
    //resizeWindow("Right Eye Image", img2.cols/4, img2.rows/4);
    //imshow("Right Eye Image", img2);    
    //waitKey(0);

    //// Stereo: disparity map
    //cout << "Start stereo matching..." << endl;
    //stereo_match(img1, img2, disp);
    //namedWindow("Disparity Map", WINDOW_NORMAL);
    //imshow("Disparity Map", disp);
    //cout << "Done matching" << endl;
    //waitKey(0);

    // ToDo: Disparity Map

    // --------- SfM Implementation --------------
    Mat imgL = imread("../SourceImages/L1.jpg");
    Mat imgR = imread("../SourceImages/M1.jpg");
    structureFromMotion(imgL, imgR);
   
    return 0;
}

// ----------------------------------
// ---- Function Implementation -----
// ----------------------------------
void structureFromMotion(Mat &img1, Mat &img2)
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

void blurDistant( Mat& img1, Mat& img2, Mat& disparity) {
    cout << "inside blurDistant" << endl;
    Mat image_blurred;
    stereo_match(img1,img2,disparity);
    cout << "Image size is " << img1.rows << " and " << img1.cols << endl;
    imshow("Blur Distant - Disparity image", disparity);
    GaussianBlur(img1, image_blurred, Size(15,15), 0);
    cv::Size s = disparity.size();
    cout << "disparity matrix is " << s.height << " and " << s.width << endl;
    for (int i = 0; i < img1.rows-3; i++) {
        cout << i << endl;
        for (int j = 0; j < img1.cols-3; j++) {
            if (disparity.at<int>(i, j) >= 0 && disparity.at <int> (i, j) <= 10) {
                image_blurred.at<cv::Vec3b>(i, j)[0] = img1.at<cv::Vec3b>(i, j)[0];
                image_blurred.at<cv::Vec3b>(i, j)[1] = img1.at<cv::Vec3b>(i, j)[1] ;
                image_blurred.at<cv::Vec3b>(i, j)[2] = img1.at<cv::Vec3b>(i, j)[2] ;
            }
        }
    }
    namedWindow("Blurred imaged based on depth", WINDOW_NORMAL);
    imshow("Blurred imaged based on depth", image_blurred);
    waitKey();
}


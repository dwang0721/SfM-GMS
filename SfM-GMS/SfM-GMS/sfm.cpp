#include "sfm.h"
int alg = STEREO_SGBM;

// learning reference:
// https://jayrambhia.com/blog/disparity-mpas

// Tasks:
// 1. parser and image input 
// 2. LOGOS application for SfM
// 3. GMS application for SfM
// 4. baseline implementation

// main - a quick test of OpenCV			
int main(int argc, char* argv[])
{
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

    Ptr<SIFT> detector = SIFT::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptor1, descriptor2;
    // Obtain keypoints and descriptors
    detector->detectAndCompute(img1, noArray(), keypoints1, descriptor1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptor2);
    Ptr<BFMatcher> matcher1 = BFMatcher::create();
    vector<DMatch> matches, matchesGMS;
    // Match two images' descriptors
    matcher1->match(descriptor1, descriptor2, matches);
    // GMS
    cv::xfeatures2d::matchGMS(img1.size(), img2.size(), keypoints1, keypoints2, matches, matchesGMS);

    Ptr<FlannBasedMatcher> matcher2 = FlannBasedMatcher::create();
    BOWKMeansTrainer bow(50);
    Mat dict = bow.cluster(descriptor1);
    vector<int> nn1, nn2;
    vector<DMatch> m1, m2, logosMatches;
    matcher2->add(dict);
    matcher2->match(descriptor1, m1);
    matcher2->match(descriptor2, m2);

    for (auto m : m1) {
        nn1.push_back(m.trainIdx);
    }
    for (auto m : m2) {
        nn2.push_back(m.trainIdx);
    }
    // LOGOS
    cv::xfeatures2d::matchLOGOS(keypoints1, keypoints2, nn1, nn2, logosMatches);


    Mat image_show1;
    drawMatches(img1, keypoints1, img2, keypoints2, matchesGMS, image_show1);
    namedWindow("GMS", WINDOW_NORMAL);
    float SCALE = 1.0;
    // Scale down the window size
    resizeWindow("GMS", image_show1.cols / SCALE, image_show1.rows / SCALE);
    imshow("GMS", image_show1);
    waitKey(0);
    
    Mat image_show2;
    drawMatches(img1, keypoints1, img2, keypoints2, logosMatches, image_show2, Scalar::all(-1),
        Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    namedWindow("LOGOS", WINDOW_NORMAL);
    resizeWindow("LOGOS", image_show2.cols / SCALE, image_show2.rows / SCALE);  // SCALE = 32 worked well for my system
    imshow("LOGOS", image_show2);
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
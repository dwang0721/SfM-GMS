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


    // ------------- Disparity Map --------------------  
    const Mat img5 = imread("../SourceImages/Disparity_L.jpg");
    const Mat img6 = imread("../SourceImages/Disparity_R.jpg");
    Ptr<SIFT> detector = SIFT::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptor1, descriptor2;
    Mat image_show;
    // Obtain keypoints and descriptors
    detector->detectAndCompute(img5, noArray(), keypoints1, descriptor1);
    detector->detectAndCompute(img6, noArray(), keypoints2, descriptor2);
    Ptr<BFMatcher> matcher = BFMatcher::create();
    vector<DMatch> matches, matchesGMS;
    // Match two images' descriptors
    // matcher->match(descriptor1, descriptor2, matches);
    cv::xfeatures2d::matchGMS(img5.size(), img6.size(), keypoints1, keypoints2, matches, matchesGMS);    
    drawMatches(img5, keypoints1, img6, keypoints2, matchesGMS, image_show);
    namedWindow("Match Image", WINDOW_NORMAL);
    float SCALE = 1.0;
    // Scale down the window size
    resizeWindow("Match Image", image_show.cols / SCALE, image_show.rows / SCALE);
    imshow("Match Image", image_show);
    waitKey(0);

    // --------- SfM Implementation --------------
    //Mat imgL = imread("../SourceImages/L1.jpg");
    //Mat imgR = imread("../SourceImages/M1.jpg");
    //structureFromMotion(imgL, imgR);
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


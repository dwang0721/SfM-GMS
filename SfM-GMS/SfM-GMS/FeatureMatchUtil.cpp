#include "FeatureMatchUtil.h"

/**
*	Using SFIT to detect key points on input image.
*	@param img, input image.
*	@param kpts, Key Points.
*   @param desc, descriptors.
*/
void SIFTDetectAndCompute(Mat& img, vector<KeyPoint>& kpts, Mat& desc) {
    Ptr<Feature2D> sift = SIFT::create(10000);
    sift->detectAndCompute(img, Mat(), kpts, desc);
}

/**
*	brute force matching descriptors in two images.
*	@param desc1, descriptor 1.
*	@param desc2, descriptor 2.
*   @param matches, result of matches.
*/
void bruteForceMatch(Mat& desc1, Mat& desc2, vector<DMatch>& matches) {
    matches.clear();
    BFMatcher desc_matcher(cv::NORM_L2, true);
    desc_matcher.match(desc1, desc2, matches, Mat());
    sort(matches.begin(), matches.end());
    while (matches.front().distance * kDistanceCoef < matches.back().distance) {
        matches.pop_back();
    }
    while (matches.size() > kMaxMatchingSize) {
        matches.pop_back();
    }
}

inline void SIFT_detect_and_compute(Mat& img, vector<KeyPoint>& kpts, Mat& desc) {
    Ptr<Feature2D> sift = SIFT::create(); // create SIFT detector
    sift->detectAndCompute(img, Mat(), kpts, desc); // call detect and compute function to get keypoints and descriptors
}

inline void match(Mat& desc1, Mat& desc2, vector<DMatch>& matches, double kDistanceCoef, int kMaxMatchingSize) {
    matches.clear();
    Ptr<BFMatcher> desc_matcher = BFMatcher::create(); // create bruteforce matcher
    desc_matcher->cv::DescriptorMatcher::match(desc1, desc2, matches, Mat()); // calculate matches

    std::sort(matches.begin(), matches.end()); //sort matches
    while (matches.front().distance * kDistanceCoef < matches.back().distance) { // eliminate some matches based on distance
        matches.pop_back();
    }
    while (matches.size() > kMaxMatchingSize) { // eliminate excess matches
        matches.pop_back();
    }
}

void SIFT_matchGMA(Mat& img1, Mat& img2, vector<DMatch>& matchesGMS, bool draw_result){
    Mat descriptor1, descriptor2;
    vector<KeyPoint> keypoints1, keypoints2;
    SIFTDetectAndCompute(img1, keypoints1, descriptor1);
    SIFTDetectAndCompute(img2, keypoints2, descriptor2);

    Ptr<BFMatcher> matcherBF = BFMatcher::create();
    vector<DMatch> matches;
    matcherBF->match(descriptor1, descriptor2, matches);
    // GMS
    cv::xfeatures2d::matchGMS(img1.size(), img2.size(), keypoints1, keypoints2, matches, matchesGMS);

    if (draw_result){
        Mat image_show1;
        drawMatches(img1, keypoints1, img2, keypoints2, matchesGMS, image_show1);
        namedWindow("GMS", WINDOW_NORMAL);
        // Scale down the window size
        resizeWindow("GMS", image_show1.cols / 2, image_show1.rows / 2);
        imshow("GMS", image_show1);
        waitKey(0);
    }
}

void SIFT_matchLOGOS(Mat& img1, Mat& img2, vector<DMatch>& matchesLOGOS, bool draw_result){
    Mat descriptor1, descriptor2;
    vector<KeyPoint> keypoints1, keypoints2;
    SIFTDetectAndCompute(img1, keypoints1, descriptor1);
    SIFTDetectAndCompute(img2, keypoints2, descriptor2);

    Ptr<FlannBasedMatcher> matcher2 = FlannBasedMatcher::create();
    BOWKMeansTrainer bow(50);
    Mat dict = bow.cluster(descriptor1);
    vector<int> nn1, nn2;
    vector<DMatch> m1, m2;
    matcher2->add(dict);
    matcher2->match(descriptor1, m1);
    matcher2->match(descriptor2, m2);

    for (auto m : m1) {
        nn1.push_back(m.trainIdx);
    }
    for (auto m : m2) {
        nn2.push_back(m.trainIdx);
    }

    xfeatures2d::matchLOGOS(keypoints1, keypoints2, nn1, nn2, matchesLOGOS);

    if (draw_result){
        Mat image_show2;
        drawMatches(img1, keypoints1, img2, keypoints2, matchesLOGOS, image_show2, Scalar::all(-1),
            Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        namedWindow("LOGOS", WINDOW_NORMAL);
        resizeWindow("LOGOS", image_show2.cols / 2, image_show2.rows / 2);  // SCALE = 32 worked well for my system
        imshow("LOGOS", image_show2);
        waitKey(0);
    }
}
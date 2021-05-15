#include "FeatureMatchUtil.h"

/**
*	Using SFIT to detect key points on input image.
*	@param img, input image.
*	@param kpts, Key Points.
*   @param desc, descriptors.
*/
void SIFTDetectAndCompute(Mat& img, vector<KeyPoint>& kpts, Mat& desc) {
    Ptr<Feature2D> sift = SIFT::create();
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

void match_LOGOS(Mat& desc1, Mat& desc2, vector<KeyPoint>& kp1,
	vector<KeyPoint>& kp2, vector<DMatch>& logos_matches)
{
	vector<int> nn1, nn2;

	BOWKMeansTrainer bow(num_words);
	Mat dict = bow.cluster(desc1);
	vector<DMatch> m1, m2, logosMatches;
	Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
	matcher->add(dict);
	matcher->match(desc1, m1);
	matcher->match(desc2, m2);

	for (auto m : m1) {
		nn1.push_back(m.trainIdx);
	}
	for (auto m : m2) {
		nn2.push_back(m.trainIdx);
	}

	cv::xfeatures2d::matchLOGOS(kp1, kp2, nn1, nn2, logos_matches);
}
#include "sfm.h"
//#include "FeatureMatchUtil.h"
int alg = STEREO_SGBM;
const int num_words = 100;
// Tasks:
// 1. parser and image input
// 2. LOGOS application for SfM <-Purrnima
// 3. GMS application for SfM  <-- Neil
// 4. baseline implementation
//     a. how to use match points for disparity map <--- Ahmed
//     b. how to use match points for reconstruction 3d. <--- Di

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

void preprocessFrame(cv::Mat& frame) {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame, frame);
    frame.convertTo(frame, CV_64FC3, 1.0 / 255.0);
    cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0);
}

// Shifts image by x,y towards right and bottom
cv::Mat shift(cv::Mat& frame, int x = 0, int y = 0) {
    int rows = frame.rows, cols = frame.cols;
    cv::Mat out = cv::Mat::zeros(frame.size(), frame.type());
    frame(cv::Rect(0, 0, frame.cols - x, frame.rows - y)).copyTo(out(cv::Rect(x, y, frame.cols - x, frame.rows - y)));
    return out;
}

cv::Mat imfilter(cv::Mat& frame, cv::Mat& kernel) {
    cv::Mat result = cv::Mat::zeros(frame.size(), frame.type());
    cv::filter2D(frame, result, -1, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
    return result;
}

// algorithm proceeds as per the following tutorial
cv::Mat calcDisparityMap(int k_size, int max_disp, cv::Mat& fr_left, cv::Mat& fr_right) {

    // define kernel with appropriate size
    int size_ = 2 * k_size + 1;
    cv::Mat kernel = cv::Mat::ones(size_, size_, CV_32F) / (float)(size_);

    // calculate SAD for all levels of shift
    std::vector<cv::Mat> maps_(max_disp);
    for (int k = 1; k <= max_disp; k++) {
        cv::Mat r_shifted = shift(fr_right, k);
        cv::Mat r_diff; cv::absdiff(fr_left, r_shifted, r_diff);
        cv::Mat r_filtered = imfilter(r_diff, kernel);
        maps_[k - 1] = r_filtered;
    }

    // calculate final disparity map by calculating minimums
    cv::Mat result = fr_left;

    double max_val = (double)INT_MIN;
    for (int row = 0; row < fr_left.rows; row++) {
        for (int col = 0; col < fr_left.cols; col++) {
            double min_val = (double)INT_MAX;
            for (int k = 0; k < max_disp; k++)
                min_val = std::min(min_val, maps_[k].at<double>(row, col));
            result.at<double>(row, col) = min_val;
            max_val = std::max(min_val, max_val);
        }
    }
    //normalize(result, result, 0, 255, NORM_MINMAX, CV_8U);
    return result;
}

/*void setFrames(cv::Mat left, cv::Mat right) {
    fr_left = left;
    fr_right = right;
}*/

cv::Mat getDisparityMap(int k_size, int max_disp, cv::Mat& fr_left, cv::Mat& fr_right) {
    // error checking block
    if (fr_left.cols != fr_right.cols || fr_left.rows != fr_right.rows)
        throw std::invalid_argument("Images are not the same size");
    else {
        preprocessFrame(fr_left);
        preprocessFrame(fr_right);
        return calcDisparityMap(k_size, max_disp, fr_left, fr_right);
    }
}

// ----------------------------------
// ---- Function Implementation -----
// ----------------------------------
void processParser(CommandLineParser parser) {
    if (parser.has("help"))
    {
        printHelp();
    }

    if (parser.has("algorithm"))
    {
        std::string _alg = parser.get<string>("algorithm");
        if (_alg == "SGBM") {
            alg = STEREO_SGBM;
        }
        // continue other algorithm
    }
}

void printHelp() {
    printf("\nhelp message here!");
}

void stereo_match(const Mat& img1, const Mat& img2, Mat& disparity) {
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

void blurDistant(Mat& img1, Mat& img2, Mat& disparity) {
    cout << "inside blurDistant" << endl;
    Mat image_blurred;
    Mat img1_clone = img1.clone();
    Mat img2_clone = img2.clone();
    disparity = getDisparityMap(4, 64, img1_clone, img2_clone);
    //stereo_match(img1,img2,disparity);
    cout << "Image size is " << img1.rows << " and " << img1.cols << endl;
    //imshow("Blur Distant - Disparity image", disparity);
    GaussianBlur(img1, image_blurred, Size(15, 15), 0);
    cv::Size s = disparity.size();
    cout << "disparity matrix is " << s.height << " and " << s.width << endl;
    for (int i = 0; i < img1.rows - 3; i++) {
        //cout << i << endl;
        for (int j = 0; j < img1.cols - 3; j++) {
            if (disparity.at<int>(i, j) >= 0 && disparity.at <int>(i, j) <= 10) {
                image_blurred.at<cv::Vec3b>(i, j)[0] = img1.at<cv::Vec3b>(i, j)[0];
                image_blurred.at<cv::Vec3b>(i, j)[1] = img1.at<cv::Vec3b>(i, j)[1];
                image_blurred.at<cv::Vec3b>(i, j)[2] = img1.at<cv::Vec3b>(i, j)[2];
            }
        }
    }
    namedWindow("Blurred imaged based on depth", WINDOW_NORMAL);
    imshow("Blurred imaged based on depth", image_blurred);
    waitKey();
}
//====================================================================================================================================================================================================

void match_LOGOS1(Mat& desc1, Mat& desc2, vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, vector<DMatch>& logos_matches) {
    vector<int> nn1, nn2;

    BOWKMeansTrainer bow(num_words);
    Mat dict = bow.cluster(desc1);
    vector<DMatch> m1, m2, logosMatches;
    Ptr<FlannBasedMatcher> matcher = /*FlannBasedMatcher::create()*/new FlannBasedMatcher(new flann::KDTreeIndexParams(4));
    //Ptr<FlannBasedMatcher> matcher = new FlannBasedMatcher(new flann::KDTreeIndexParams(4));
    matcher->add(dict);
    matcher->match(desc1, m1);
    matcher->match(desc2, m2);
    cout << "Matches size is " << logos_matches.size() << endl;

    for (auto m : m1) {
        nn1.push_back(m.trainIdx);
    }
    for (auto m : m2) {
        nn2.push_back(m.trainIdx);
    }

    cv::xfeatures2d::matchLOGOS(kp1, kp2, nn1, nn2, logos_matches);
    cout << "Matches size is " << logos_matches.size() << endl;
}

void disp_calculate(Mat img_1, Mat img_2, Mat gt, String str, String out_dir, int  disp_ratio, float ratio, String disparity_type)
{

    unsigned long start_time = 0, finish_time = 0; //check processing time  
    start_time = getTickCount(); //check processing time
    Ptr<Feature2D> f2d;
    Ptr<DescriptorMatcher> matcher;
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    if (str == "sift")
    {
        f2d = SIFT::create();
        matcher = new FlannBasedMatcher(new flann::KDTreeIndexParams(4));
    }
    else if (str == "orb")
    {
        f2d = ORB::create();
        matcher = new FlannBasedMatcher(new flann::LshIndexParams(5, 20, 2)); //Using ORB, Flann-based Locality-sensitive hashing(LSH) for matching:

    }

    if (disparity_type == "dense")
    {
        for (int i = 0; i < img_1.cols; i++)
            for (int j = 0; j < img_1.rows; j++) //Dense disparity
            {
                keypoints_1.push_back(KeyPoint(i, j, 1));
                keypoints_2.push_back(KeyPoint(i, j, 1));
            }
        f2d->compute(img_1, keypoints_1, descriptors_1); // To get descriptors at each and every pixel(dense disparity)
        f2d->compute(img_2, keypoints_2, descriptors_2); // To get descriptors at each and every pixel(dense disparity)
    }

    else if (disparity_type == "sparse")
    {
        f2d->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1); //Sparse Disparity
        f2d->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2); //Sparse Disparity
    }
    vector< DMatch > matches;
    matcher->match(descriptors_1, descriptors_2, matches);

    // Calculating best matches of obtained matches by using Lowe's Ratio:
    double max_dist = 0; double min_dist = 100;
    for (int i = 0; i < matches.size(); i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    //printf("\n-- Max dist : %f ", max_dist);
    //printf("\n-- Min dist : %f ", min_dist);
    std::vector< DMatch > good_matches;
    vector<Point2f>imgpts1, imgpts2;
    for (int i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance <= max(ratio * max_dist, 0.02)) {
            good_matches.push_back(matches[i]);
            imgpts1.push_back(keypoints_1[matches[i].queryIdx].pt);
            imgpts2.push_back(keypoints_2[matches[i].trainIdx].pt);
        }
    }

    Mat img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    namedWindow("Good Matches_" + str, cv::WINDOW_AUTOSIZE);
    imshow("Good Matches_" + str, img_matches);
    imwrite(out_dir + "/Good Matches_" + str + ".png", img_matches);

    //Calculating dispairity and disparity map:
    Mat disparity(img_1.size().height, img_1.size().width, CV_8U, Scalar(255));
    for (int i = 0; i < (int)good_matches.size(); i++)
    {
        int x = keypoints_1[good_matches[i].queryIdx].pt.x;
        int y = keypoints_1[good_matches[i].queryIdx].pt.y;
        int x1 = keypoints_2[good_matches[i].trainIdx].pt.x;
        //printf("\nx point %d, y point %d,disparity-->%d", x, x1, abs(x - x1));
        //printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  ", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
        disparity.at<uchar>(y, x) = abs(x - x1);
    }

    //Calculating the RMS value of the dispairty
    long float rms = 0, count = 0, max_disp = 0;
    for (int i = 0; i < gt.size().width; i++) {
        for (int j = 0; j < gt.size().height; j++) {
            if (disparity.at<uchar>(j, i) != 255)
            {
                int a = abs(disparity.at<uchar>(j, i) - gt.at<uchar>(j, i) / disp_ratio);
                if (a > max_disp)
                    max_disp = a;
                //printf("%d, %d, %d, %d\n",disparity(j,i),gt.at<uchar>(j, i)/4, a, a*a);
                rms = rms + a * a;
                count = count + 1;
            }
        }
    }
    printf("\n No. of disparities--->%d,   rms--->%f", (int)count, sqrt(rms / count));


    //Calculating the Bad pixel % at dispairties for thresholds-1,2,5
    int delta[3] = { 1, 2, 5 };
    float bad[3] = { 0.0,0.0,0.0 };
    for (int k = 0; k < 3; k++)
    {
        int threshold_temp = delta[k];
        for (int i = 0; i < gt.size().width; i++)
            for (int j = 0; j < gt.size().height; j++)
            {
                if (disparity.at<uchar>(j, i) != 255)
                {
                    int a = abs(disparity.at<uchar>(j, i) - gt.at<uchar>(j, i) / disp_ratio);
                    //printf("%d, %d, %d\n", disparity.at<uchar>(j, i),gt.at<uchar>(j, i)/4,a );
                    if (a > threshold_temp)
                        bad[k] = bad[k] + a;
                }
            }

        printf("\n bad pixel percent(delta>%d)--->%f", threshold_temp, bad[k] / count);
    }


    //Grayscale Normalization
    for (int i = 0; i < gt.size().width; i++)
        for (int j = 0; j < gt.size().height; j++)
            if (disparity.at<uchar>(j, i) != 255)
                disparity.at<uchar>(j, i) = disparity.at<uchar>(j, i) * disp_ratio * 255 / max_disp;

    finish_time = getTickCount(); //check processing time
    printf("\nElapsed Time : %.2lf sec \n", (finish_time - start_time) / getTickFrequency());         //check processing time  

    imshow("disparity_" + str + ";    rms:" + to_string(sqrt(rms / count)) + "    bp%: " + to_string(bad[0] / count), disparity);
    imwrite(out_dir + "/disparity_" + str + ".png", disparity);
    imshow("ground Truth", gt);
}
//====================================================================================================================================================================================================

void matchBasedDispCalculate(Mat img_1, Mat img_2, Mat gt, String alg, /*String out_dir,*/ int  disp_ratio, float ratio, String disparity_type)
{
    unsigned long start_time = 0, finish_time = 0; //check processing time  
    start_time = getTickCount(); //check processing time
    Ptr<Feature2D> f2d;
    Ptr<DescriptorMatcher> matcher;
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    if (alg == "sift")
    {
        f2d = SIFT::create();
        matcher = new FlannBasedMatcher(new flann::KDTreeIndexParams(4));
    }
    else if (alg == "orb")
    {
        f2d = ORB::create();
        matcher = new FlannBasedMatcher(new flann::LshIndexParams(5, 20, 2)); //Using ORB, Flann-based Locality-sensitive hashing(LSH) for matching:
    }
    else if (alg == "GMS")
    {
        f2d = SIFT::create();
        //vector<KeyPoint> keypoints1, keypoints2;
        //Mat descriptor1, descriptor2;
        // Obtain keypoints and descriptors
        //detector->detectAndCompute(img_1, noArray(), keypoints_1, descriptors_1);
        //detector->detectAndCompute(img_2, noArray(), keypoints_2, descriptors_2);
        matcher = /*BFMatcher::create(2,false);*/new FlannBasedMatcher(new flann::KDTreeIndexParams(4));
        //matcher = DescriptorMatcher::create("BruteForce-Hamming");
        //vector<DMatch> matches, matchesGMS;
        // Match two images' descriptors
        //matcher->match(descriptors_1, descriptors_2, matches);
        //cv::xfeatures2d::matchGMS(img_1.size(), img_2.size(), keypoints_1, keypoints_2, matches, matchesGMS);
        //Mat image_show;
        //drawMatches(img_1, keypoints_1, img_2, keypoints_2, matchesGMS, image_show);
    }
    else if (alg == "LOGOS")
    {
        cout << "LOGOS" << endl;
        const int nFeatures = 10;
        f2d = SIFT::create(/*nFeatures*/);
    }

    if (disparity_type == "dense")
    {
        for (int i = 0; i < img_1.cols; i++)
            for (int j = 0; j < img_1.rows; j++) //Dense disparity
            {
                keypoints_1.push_back(KeyPoint(i, j, 1));
                keypoints_2.push_back(KeyPoint(i, j, 1));
            }
        f2d->compute(img_1, keypoints_1, descriptors_1); // To get descriptors at each and every pixel(dense disparity)
        f2d->compute(img_2, keypoints_2, descriptors_2); // To get descriptors at each and every pixel(dense disparity)
    }

    else if (disparity_type == "sparse")
    {
        f2d->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1); //Sparse Disparity
        f2d->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2); //Sparse Disparity
    }
    vector< DMatch > matches;
    if (alg.compare("LOGOS") != 0) {
        matcher->match(descriptors_1, descriptors_2, matches);
    }

    if (alg == "GMS") {
        cout << "applying GMS to keypoints" << endl;
        cout << "Matches size is " << matches.size() << endl;
        vector<DMatch> matchesGMS;
        cv::xfeatures2d::matchGMS(img_1.size(), img_2.size(), keypoints_1, keypoints_2, matches, matches/*GMS*/);
        cout << "Matches size is " << matches.size() << endl;
        /*Mat image_show;
        drawMatches(img_1, keypoints_1, img_2, keypoints_2, matchesGMS, image_show);
        namedWindow("GMS Match Image", WINDOW_NORMAL);
        float SCALE = 1.0;
        // Scale down the window size
        resizeWindow("GMS Match Image", image_show.cols / SCALE, image_show.rows / SCALE);
        imshow("GMS Match Image", image_show);
        //waitKey(0);*/
    }

    if (alg == "LOGOS") {
        cout << "applying LOGOS to keypoints" << endl;
        cout << "Matches size is " << matches.size() << endl;
        match_LOGOS1(descriptors_1, descriptors_2, keypoints_1, keypoints_2, matches);
        Mat output;
        cout << "Matches size is " << matches.size() << endl;
        /*drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, output, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        const int SCALE = 4;
        namedWindow("LOGOS output", WINDOW_NORMAL);
        resizeWindow("LOGOS output", output.cols / SCALE, output.rows / SCALE);
        imshow("LOGOS output", output);
        waitKey(0);*/
    }

    // Calculating best matches of obtained matches by using Lowe's Ratio:
    double max_dist = 0; double min_dist = 100;
    for (int i = 0; i < matches.size(); i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    //printf("\n-- Max dist : %f ", max_dist);
    //printf("\n-- Min dist : %f ", min_dist);
    std::vector< DMatch > good_matches;
    vector<Point2f>imgpts1, imgpts2;
    for (int i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance <= max(ratio * max_dist, 0.02)) {
            good_matches.push_back(matches[i]);
            imgpts1.push_back(keypoints_1[matches[i].queryIdx].pt);
            imgpts2.push_back(keypoints_2[matches[i].trainIdx].pt);
        }
    }

    cout << "Good Matches size is " << good_matches.size() << endl;

    Mat img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    namedWindow("Good Matches_" + alg, cv::WINDOW_AUTOSIZE);
    imshow("Good Matches_" + alg, img_matches);
    //waitKey();
    //imwrite(out_dir + "/Good Matches_" + str + ".png", img_matches);


    //Calculating dispairity and disparity map:
    Mat disparity(img_1.size().height, img_1.size().width, CV_8U, Scalar(/*255*/0));
    for (int i = 0; i < (int)good_matches.size(); i++)
    {
        int x = keypoints_1[good_matches[i].queryIdx].pt.x; // good matches index of img1 descriptor, after the index substitute into the keypoints of img1 then get the x-coordinates of the keypoint
        int y = keypoints_1[good_matches[i].queryIdx].pt.y; // good matches index of img1 descriptor, after the index substitute into the keypoints of img1 then get the y-coordinates of the keypoint
        int x1 = keypoints_2[good_matches[i].trainIdx].pt.x; // good matches index of img2 descriptor, after the index substitute into the keypoints of img1 then get the x - coordinates of the keypoint
        //printf("\nx point %d, y point %d,disparity-->%d", x, x1, abs(x - x1));
        //printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  ", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
        disparity.at<uchar>(y, x) = abs(x - x1);
    }

    //Calculating the RMS value of the dispairty
    long float rms = 0, count = 0, max_disp = 0;
    for (int i = 0; i < gt.size().width; i++) {
        for (int j = 0; j < gt.size().height; j++) {
            if (disparity.at<uchar>(j, i) != 255) // disparity is not white 255
            {
                int a = abs(disparity.at<uchar>(j, i) - gt.at<uchar>(j, i) / disp_ratio);
                if (a > max_disp)
                    max_disp = a;
                //printf("%d, %d, %d, %d\n",disparity(j,i),gt.at<uchar>(j, i)/4, a, a*a);
                rms = rms + a * a;
                count = count + 1;
            }
        }
    }
    printf("\n No. of disparities--->%d,   rms--->%f", (int)count, sqrt(rms / count));


    //Calculating the Bad pixel % at dispairties for thresholds-1,2,5
    int delta[3] = { 1, 2, 5 };
    float bad[3] = { 0.0,0.0,0.0 };
    for (int k = 0; k < 3; k++)
    {
        int threshold_temp = delta[k];
        for (int i = 0; i < gt.size().width; i++)
            for (int j = 0; j < gt.size().height; j++)
            {
                if (disparity.at<uchar>(j, i) != 255)
                {
                    int a = abs(disparity.at<uchar>(j, i) - gt.at<uchar>(j, i) / disp_ratio);
                    //printf("%d, %d, %d\n", disparity.at<uchar>(j, i),gt.at<uchar>(j, i)/4,a );
                    if (a > threshold_temp)
                        bad[k] = bad[k] + a;
                }
            }

        printf("\n bad pixel percent(delta>%d)--->%f", threshold_temp, bad[k] / count);
    }


    //Grayscale Normalization
    for (int i = 0; i < gt.size().width; i++)
        for (int j = 0; j < gt.size().height; j++)
            if (disparity.at<uchar>(j, i) != 255)
                disparity.at<uchar>(j, i) = disparity.at<uchar>(j, i) * disp_ratio * 255 / max_disp;

    finish_time = getTickCount(); //check processing time
    printf("\nElapsed Time : %.2lf sec \n", (finish_time - start_time) / getTickFrequency());         //check processing time  

    imshow("disparity_" + alg + ";    rms:" + to_string(sqrt(rms / count)) + "    bp%: " + to_string(bad[0] / count), disparity);
    //waitKey();
    //imwrite(out_dir + "/disparity_" + str + ".png", disparity);
    //imshow("ground Truth", gt);
}

//====================================================================================================================================================================================================

void disp_calculate_matches_nogt(Mat img_1, Mat img_2, /*Mat gt,*/ String alg, /*String out_dir,*/ int  disp_ratio, float ratio, String disparity_type)
{
    unsigned long start_time = 0, finish_time = 0; //check processing time  
    start_time = getTickCount(); //check processing time
    Ptr<Feature2D> f2d;
    Ptr<DescriptorMatcher> matcher;
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    if (alg == "sift")
    {
        f2d = SIFT::create();
        matcher = new FlannBasedMatcher(new flann::KDTreeIndexParams(4));
    }
    else if (alg == "orb")
    {
        f2d = ORB::create();
        matcher = new FlannBasedMatcher(new flann::LshIndexParams(5, 20, 2)); //Using ORB, Flann-based Locality-sensitive hashing(LSH) for matching:
    }
    else if (alg == "GMS")
    {
        f2d = SIFT::create();
        //vector<KeyPoint> keypoints1, keypoints2;
        //Mat descriptor1, descriptor2;
        // Obtain keypoints and descriptors
        //detector->detectAndCompute(img_1, noArray(), keypoints_1, descriptors_1);
        //detector->detectAndCompute(img_2, noArray(), keypoints_2, descriptors_2);
        matcher = /*BFMatcher::create(2,false);*/new FlannBasedMatcher(new flann::KDTreeIndexParams(4));
        //matcher = DescriptorMatcher::create("BruteForce-Hamming");
        //vector<DMatch> matches, matchesGMS;
        // Match two images' descriptors
        //matcher->match(descriptors_1, descriptors_2, matches);
        //cv::xfeatures2d::matchGMS(img_1.size(), img_2.size(), keypoints_1, keypoints_2, matches, matchesGMS);
        //Mat image_show;
        //drawMatches(img_1, keypoints_1, img_2, keypoints_2, matchesGMS, image_show);
    }
    else if (alg == "LOGOS")
    {
        cout << "LOGOS" << endl;
        const int nFeatures = 10;
        f2d = SIFT::create(/*nFeatures*/);
    }

    if (disparity_type == "dense")
    {
        for (int i = 0; i < img_1.cols; i++)
            for (int j = 0; j < img_1.rows; j++) //Dense disparity
            {
                keypoints_1.push_back(KeyPoint(i, j, 1));
                keypoints_2.push_back(KeyPoint(i, j, 1));
            }
        f2d->compute(img_1, keypoints_1, descriptors_1); // To get descriptors at each and every pixel(dense disparity)
        f2d->compute(img_2, keypoints_2, descriptors_2); // To get descriptors at each and every pixel(dense disparity)
    }

    else if (disparity_type == "sparse")
    {
        f2d->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1); //Sparse Disparity
        f2d->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2); //Sparse Disparity
    }
    vector< DMatch > matches;
    if (alg.compare("LOGOS") != 0) {
        matcher->match(descriptors_1, descriptors_2, matches);
    }

    if (alg == "GMS") {
        cout << "applying GMS to keypoints" << endl;
        cout << "Matches size is " << matches.size() << endl;
        vector<DMatch> matchesGMS;
        cv::xfeatures2d::matchGMS(img_1.size(), img_2.size(), keypoints_1, keypoints_2, matches, matches/*GMS*/);
        cout << "Matches size is " << matches.size() << endl;
        /*Mat image_show;
        drawMatches(img_1, keypoints_1, img_2, keypoints_2, matchesGMS, image_show);
        namedWindow("GMS Match Image", WINDOW_NORMAL);
        float SCALE = 1.0;
        // Scale down the window size
        resizeWindow("GMS Match Image", image_show.cols / SCALE, image_show.rows / SCALE);
        imshow("GMS Match Image", image_show);
        //waitKey(0);*/
    }

    if (alg == "LOGOS") {
        cout << "applying LOGOS to keypoints" << endl;
        cout << "Matches size is " << matches.size() << endl;
        match_LOGOS1(descriptors_1, descriptors_2, keypoints_1, keypoints_2, matches);
        Mat output;
        cout << "Matches size is " << matches.size() << endl;
        /*drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, output, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        const int SCALE = 4;
        namedWindow("LOGOS output", WINDOW_NORMAL);
        resizeWindow("LOGOS output", output.cols / SCALE, output.rows / SCALE);
        imshow("LOGOS output", output);
        waitKey(0);*/
    }

    // Calculating best matches of obtained matches by using Lowe's Ratio:
    double max_dist = 0; double min_dist = 100;
    for (int i = 0; i < matches.size(); i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    //printf("\n-- Max dist : %f ", max_dist);
    //printf("\n-- Min dist : %f ", min_dist);
    std::vector< DMatch > good_matches;
    vector<Point2f>imgpts1, imgpts2;
    for (int i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance <= max(ratio * max_dist, 0.02)) {
            good_matches.push_back(matches[i]);
            imgpts1.push_back(keypoints_1[matches[i].queryIdx].pt);
            imgpts2.push_back(keypoints_2[matches[i].trainIdx].pt);
        }
    }

    cout << "Good Matches size is " << good_matches.size() << endl;

    Mat img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    namedWindow("Good Matches_" + alg, cv::WINDOW_AUTOSIZE);
    imshow("Good Matches_" + alg, img_matches);
    //waitKey();
    //imwrite(out_dir + "/Good Matches_" + str + ".png", img_matches);


    //Calculating dispairity and disparity map:
    Mat disparity(img_1.size().height, img_1.size().width, CV_8U, Scalar(/*255*/0));
    for (int i = 0; i < (int)good_matches.size(); i++)
    {
        int x = keypoints_1[good_matches[i].queryIdx].pt.x; // good matches index of img1 descriptor, after the index substitute into the keypoints of img1 then get the x-coordinates of the keypoint
        int y = keypoints_1[good_matches[i].queryIdx].pt.y; // good matches index of img1 descriptor, after the index substitute into the keypoints of img1 then get the y-coordinates of the keypoint
        int x1 = keypoints_2[good_matches[i].trainIdx].pt.x; // good matches index of img2 descriptor, after the index substitute into the keypoints of img1 then get the x - coordinates of the keypoint
        //printf("\nx point %d, y point %d,disparity-->%d", x, x1, abs(x - x1));
        //printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  ", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
        disparity.at<uchar>(y, x) = abs(x - x1);
    }

    //Calculating the RMS value of the dispairty

    long float rms = 0, count = 0, max_disp = 0;
    for (int i = 0; i < img_1.size().width; i++) {
        for (int j = 0; j < img_1.size().height; j++) {
            if (disparity.at<uchar>(j, i) != 255) // disparity is not white 255
            {
                int a = abs(disparity.at<uchar>(j, i) - img_1.at<uchar>(j, i) / disp_ratio);
                if (a > max_disp)
                    max_disp = a;
                //printf("%d, %d, %d, %d\n",disparity(j,i),gt.at<uchar>(j, i)/4, a, a*a);
                rms = rms + a * a;
                count = count + 1;
            }
        }
    }
    printf("\n No. of disparities--->%d,   rms--->%f", (int)count, sqrt(rms / count));

    /*
    //Calculating the Bad pixel % at dispairties for thresholds-1,2,5
    int delta[3] = { 1, 2, 5 };
    float bad[3] = { 0.0,0.0,0.0 };
    for (int k = 0; k < 3; k++)
    {
    int threshold_temp = delta[k];
    for (int i = 0; i < gt.size().width; i++)
    for (int j = 0; j < gt.size().height; j++)
    {
    if (disparity.at<uchar>(j, i) != 255)
    {
    int a = abs(disparity.at<uchar>(j, i) - gt.at<uchar>(j, i) / disp_ratio);
    //printf("%d, %d, %d\n", disparity.at<uchar>(j, i),gt.at<uchar>(j, i)/4,a );
    if (a > threshold_temp)
    bad[k] = bad[k] + a;
    }
    }

    printf("\n bad pixel percent(delta>%d)--->%f", threshold_temp, bad[k] / count);
    }
    */

    //Grayscale Normalization
    for (int i = 0; i < img_1.size().width; i++)
        for (int j = 0; j < img_1.size().height; j++)
            if (disparity.at<uchar>(j, i) != 255)
                disparity.at<uchar>(j, i) = disparity.at<uchar>(j, i) * disp_ratio * 255 / max_disp;

    finish_time = getTickCount(); //check processing time
    printf("\nElapsed Time : %.2lf sec \n", (finish_time - start_time) / getTickFrequency());         //check processing time  

    imshow("disparity_" + alg + ";    rms:" + to_string(sqrt(rms / count))/* + "    bp%: " + to_string(bad[0] / count)*/, disparity);
    //waitKey();
    //imwrite(out_dir + "/disparity_" + str + ".png", disparity);
    //imshow("ground Truth", gt);
}
//====================================================================================================================================================================================================

void test_LOGOS(Mat& image1, Mat& image2) {
    vector<DMatch> logos_matches;
    vector<KeyPoint> kp1, kp2;

    Mat descriptor1, descriptor2;
    const int nFeatures = 5000;
    Ptr<SIFT> detector = SIFT::create(nFeatures);

    detector->detectAndCompute(image1, noArray(), kp1, descriptor1);
    // Detects keypoints and computes sift descriptors for image1

    detector->detectAndCompute(image2, noArray(), kp2, descriptor2);
    // Detects keypoints and computes sift descriptors for image2

    //----------------------------------------------------------------
    match_LOGOS1(descriptor1, descriptor2, kp1, kp2, logos_matches);
    Mat output;

    drawMatches(image1, kp1, image2, kp2, logos_matches, output, Scalar::all(-1),
        Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    const int SCALE = 4;
    namedWindow("output", WINDOW_NORMAL);
    resizeWindow("output", output.cols / SCALE, output.rows / SCALE);
    imshow("output", output);
    waitKey(0);
}

// main - a quick test of OpenCV
int main(int argc, char* argv[])
{
    //GMS
    /*const Mat img5 = imread("../SourceImages/Disparity_L.jpg");
    const Mat img6 = imread("../SourceImages/Disparity_R.jpg");
    Ptr<SIFT> detector = SIFT::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptor1, descriptor2;
    // Obtain keypoints and descriptors
    detector->detectAndCompute(img5, noArray(), keypoints1, descriptor1);
    detector->detectAndCompute(img6, noArray(), keypoints2, descriptor2);
    Ptr<BFMatcher> matcher = BFMatcher::create();
    vector<DMatch> matches, matchesGMS;
    // Match two images' descriptors
    matcher->match(descriptor1, descriptor2, matches);
    Mat image_show;
    drawMatches(img5, keypoints1, img6, keypoints2, matches, image_show);
    namedWindow("Match Image without GMS", WINDOW_NORMAL);
    float SCALE = 1.0;
    // Scale down the window size
    resizeWindow("Match Image without GMS", image_show.cols / SCALE, image_show.rows / SCALE);
    imshow("Match Image without GMS", image_show);
    waitKey(0);

    cv::xfeatures2d::matchGMS(img5.size(), img6.size(), keypoints1, keypoints2, matches, matchesGMS);
    drawMatches(img5, keypoints1, img6, keypoints2, matchesGMS, image_show);
    namedWindow("GMS Match Image", WINDOW_NORMAL);
    SCALE = 1.0;
    // Scale down the window size
    resizeWindow("GMS Match Image", image_show.cols / SCALE, image_show.rows / SCALE);
    imshow("GMS Match Image", image_show);
    waitKey(0);
    */
    //LOGOS
    /*Mat image1 = imread("../SourceImages/Disparity_L.jpg");
    Mat image2 = imread("../SourceImages/Disparity_R.jpg");
    test_LOGOS(image1, image2);
    */
    // ToDo: parser implementation need
    CommandLineParser parser(argc, argv, parserKeys);
    processParser(parser);

    // initialize input

    Mat disp;

    Mat img3 = imread("../SourceImages/left.png");
    Mat img4 = imread("../SourceImages/right.png");
    /*
        disp = getDisparityMap(4, 64, img3, img4);
        namedWindow("Disparity Map - Ahmed implementation", WINDOW_NORMAL);
        imshow("Disparity Map - Ahmed implementation", disp);
        //waitKey();

        Mat disp3;
        img3 = imread("../SourceImages/left.png");
        img4 = imread("../SourceImages/right.png");
        stereo_match(img3, img4, disp3);
        namedWindow("Disparity Map - Di implementation", WINDOW_NORMAL);
        imshow("Disparity Map - Di implementation", disp3);
        waitKey(0);
    */
    //int disp_ratio = 1;
    //float ratio_sift = 0.1;
    //float ratio_orb = 0.3;
    String disparity_type = "dense";
    int disp_ratio = 4;
    String dataset = "Cones";
    if (dataset == "Cones" || dataset == "Teddy")
        disp_ratio = 4; // depends on the dataset ground truth scaling factor
    else
        disp_ratio = 1;
    float ratio_orb = 0.0;
    float ratio_sift = 0.0;
    if (disparity_type == "sparse") {
        if (dataset == "Cones") { ratio_sift = 0.4; ratio_orb = 0.4; }
        else if (dataset == "Teddy") { ratio_sift = 0.3; ratio_orb = 0.3; }
        else if (dataset == "Art") { ratio_sift = 0.1; ratio_orb = 0.3; }
    }
    else { ratio_sift = 0.1; ratio_orb = 0.1; }

    //const Mat img1 = imread("../SourceImages/Disparity_L.jpg");
    //const Mat img2 = imread("../SourceImages/Disparity_R.jpg");
    //disp_calculate_matches_nogt(img1, img2, "GMS", disp_ratio, ratio_sift, disparity_type);
    //disp_calculate_matches_nogt(img1, img2, "sift", disp_ratio, ratio_sift, disparity_type);
    //disp_calculate_matches_nogt(img1, img2, "orb", disp_ratio, ratio_orb, disparity_type);
    //waitKey();


    img3 = imread("../SourceImages/left1.png");
    img4 = imread("../SourceImages/right1.png");
    Mat gt = imread("../SourceImages/left_gt1.png");
    matchBasedDispCalculate(img3, img4, gt, "GMS", disp_ratio, ratio_sift, disparity_type);
    matchBasedDispCalculate(img3, img4, gt, "sift", disp_ratio, ratio_sift, disparity_type);
    matchBasedDispCalculate(img3, img4, gt, "orb", disp_ratio, ratio_orb, disparity_type);
    //waitKey();
    matchBasedDispCalculate(img3, img4, gt, "LOGOS", disp_ratio, ratio_sift, disparity_type);
    waitKey();
    /*
        img3 = imread("../SourceImages/view0-1.png");
        img4 = imread("../SourceImages/view2-1.png");
        Mat disp2;
        cout << "Start background blurring..." << endl;
        blurDistant(img3, img4, disp2);
        cout << "Done" << endl;
        waitKey(0);

        namedWindow("Left Eye Image", WINDOW_NORMAL);
        resizeWindow("Left Eye Image", img1.cols / 4, img1.rows / 4);
        imshow("Left Eye Image", img1);

        namedWindow("Right Eye Image", WINDOW_NORMAL);
        resizeWindow("Right Eye Image", img2.cols / 4, img2.rows / 4);
        imshow("Right Eye Image", img2);
        waitKey(0);

        // Stereo: disparity map
        cout << "Start stereo matching..." << endl;
        stereo_match(img1, img2, disp);
        namedWindow("Disparity Map", WINDOW_NORMAL);
        imshow("Disparity Map", disp);
        cout << "Done matching" << endl;
        waitKey(0);
    */
    // ToDo: SfM
    // ToDo: Disparity Map
    return 0;
}

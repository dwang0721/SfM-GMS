/**
 *
 * @author ahmed_nada, Neil_Cai, Di_Wang, Purrnima_Dyanandam
 */
 /**
  * * Class to represent the simulation of disparity map, Sparse and Dense and 
  * also basic simulation of using disparity map to create portrait mode based 
  * on stereo images.
  *
  */
#include "DisparityUtil.h"
//int alg = STEREO_SGBM;
const int num_words = 100;

/**
 * stereo match function which takes to images and returns BM (block matching)
 * disparity map
 *
 * @param input image matrixes, disparity map matrix .
 * @return VOID, fills the disparity map matrix with the disparity valuess.
 */
void stereo_match(const Mat& img1, const Mat& img2, Mat& disparity) {
	// place holder algorithm
	Mat g1, g2;
	cvtColor(img1, g1, COLOR_BGR2GRAY);
	cvtColor(img2, g2, COLOR_BGR2GRAY);

	Ptr<StereoBM> sbm = StereoBM::create(16, 5); // max disparity, window size
	sbm->setNumDisparities(224);
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
	cout << "Number of channels in disparity Mat is " << disparity.channels() << endl;
	// make the disparity image look like the result from our disparity map, swap black pixels with white
	for (int i = 0; i < disparity.cols; i++) {
		for (int j = 0; j < disparity.rows; j++) {
			if (disparity.at<uchar>(j, i) == 0) {
				disparity.at<uchar>(j, i) = 255;
			}
		}
	}
}

/**
 * match LOGOS function takes two descriptor vectors and 2 verctors of keypoints and returns 
 * the LOGOS matches
 * 
 * @param input two descriptor vectors and two keypoint vectors.
 * @return VOID, fills the LOGOS matches vector.
 */
void match_LOGOS1(Mat& desc1, Mat& desc2, vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, vector<DMatch>& logos_matches) {
	vector<int> nn1, nn2;
	BOWKMeansTrainer bow(num_words);
	cout << "BoW calculation done" << endl;
	Mat dict = bow.cluster(desc1);
	cout << "Clustering done" << endl;
	vector<DMatch> m1, m2, logosMatches;
	Ptr<FlannBasedMatcher> matcher = new FlannBasedMatcher(new flann::KDTreeIndexParams(4)); //Default # of trees 4 

	matcher->add(dict);
	matcher->match(desc1, m1);
	matcher->match(desc2, m2);

	for (auto m : m1) {
		nn1.push_back(m.trainIdx);
	}
	for (auto m : m2) {
		nn2.push_back(m.trainIdx);
	}
	cout << "Applying matchLOGOS based on calculated nn1 and nn2" << endl;
	cv::xfeatures2d::matchLOGOS(kp1, kp2, nn1, nn2, logos_matches);
}

/**
 * matchBased disparity map generation function. It takes two stereo pair of images as matrixes 
 * then it decides based on the disparity type (sparse or dense) which keypoints to work with
 * if the sparse disparity map, it uses ditectandCompute if dense it just adds keypoints as much as 
 * the number of pixels in the images then computes descriptors. SIFT and SIFT + GMS use FlannBasedMatcher
 * with 4 tree (default) and ORB uses Flann-based Locality-sensitive hashing(LSH) with default values.
 * 
 *
 *
 * @param input two descriptor vectors and two keypoint vectors.
 * @return VOID, fills the LOGOS matches vector.
 */
void matchBasedDispCalculate(Mat img_1, Mat img_2, Mat gt, String alg, int  disp_ratio, String disparity_type)
{
	unsigned long start_time = 0, finish_time = 0;
	start_time = getTickCount(); //start processing time
	Ptr<Feature2D> f2d;
	Ptr<DescriptorMatcher> matcher;
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	if (alg == "sift")
	{
		f2d = SIFT::create(); // create SIFT feature detector 
		matcher = new FlannBasedMatcher(new flann::KDTreeIndexParams(4));
	}
	else if (alg == "orb")
	{
		f2d = ORB::create(); // create ORB feature detector
		matcher = new FlannBasedMatcher(new flann::LshIndexParams(5, 20, 2)); //Locality-sensitive hashing(LSH) for matching
	}
	else if (alg == "GMS")
	{
		cout << "GMS" << endl;
		f2d = SIFT::create(); // create SIFT feature detector 
		matcher = new FlannBasedMatcher(new flann::KDTreeIndexParams(4));
	}
	else if (alg == "LOGOS")
	{
		const int nFeatures = 5000; // number of features for LOGOS
		f2d = SIFT::create(nFeatures);
	}

	if (disparity_type == "dense") //Dense disparity Map 
	{
		for (int i = 0; i < img_1.cols; i++)
			for (int j = 0; j < img_1.rows; j++) 
			{
				keypoints_1.push_back(KeyPoint(i, j, 1)); // push back keypoints equal to the number of pixels in the image rows and columns
				keypoints_2.push_back(KeyPoint(i, j, 1));
			}
		f2d->compute(img_1, keypoints_1, descriptors_1); // To get descriptors at each and every pixel(dense disparity)
		f2d->compute(img_2, keypoints_2, descriptors_2); // To get descriptors at each and every pixel(dense disparity)
	}

	else if (disparity_type == "sparse")
	{
		f2d->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1); //Sparse Disparity Map detect keypoints 
		f2d->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2); //Sparse Disparity Map detect keypoints
	}
	vector< DMatch > matches;
	if (alg.compare("LOGOS") != 0) { // for all algorithms except for LOGOS create a matcher
		cout << "applying " + alg + " to keypoints" << endl;
		matcher->match(descriptors_1, descriptors_2, matches);
	}

	if (alg == "GMS") { // for GMS, additionally eliminate outliers by running matches against GMS eliminator
		cout << "Prior GMS Matches size is " << matches.size() << endl;
		vector<DMatch> matchesGMS;
		cv::xfeatures2d::matchGMS(img_1.size(), img_2.size(), keypoints_1, keypoints_2, matches, matchesGMS);
		matches = matchesGMS;
		cout << "Post GMS Matches size is " << matches.size() << endl;
	}

	if (alg == "LOGOS") { // if LOGOS, eliminate outliers by passing the matches to LOGOS eliminator
		cout << "Applying LOGOS to keypoints" << endl;
		cout << "Prior LOGOS Matches size is " << matches.size() << endl;
		match_LOGOS1(descriptors_1, descriptors_2, keypoints_1, keypoints_2, matches);
		cout << "Post LOGOS Matches size is " << matches.size() << endl;
	}

	double max_dist = 0; double min_dist = 100;
	for (int i = 0; i < matches.size(); i++) // get min and max distance between matches
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	std::vector< DMatch > good_matches;
	good_matches = matches;

	Mat img_matches; // draw matches
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	namedWindow("Good Matches_" + alg, cv::WINDOW_AUTOSIZE);
	imshow("Good Matches_" + alg, img_matches);
	imwrite("Good Matches_" + disparity_type + "_" + alg + ".png", img_matches);

	//Calculate dispairity and disparity map
	Mat disparity(img_1.size().height, img_1.size().width, CV_8U, Scalar(255));
	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		int x = keypoints_1[good_matches[i].queryIdx].pt.x; // good matches index of img1 descriptor, after the index substitute into the keypoints of img1 then get the x-coordinates of the keypoint
		int y = keypoints_1[good_matches[i].queryIdx].pt.y; // good matches index of img1 descriptor, after the index substitute into the keypoints of img1 then get the y-coordinates of the keypoint
		int x1 = keypoints_2[good_matches[i].trainIdx].pt.x; // good matches index of img2 descriptor, after the index substitute into the keypoints of img1 then get the x - coordinates of the keypoint
		disparity.at<uchar>(y, x) = abs(x - x1); // disparity is the x-coordinate difference between pixel matches
	}

	//RMS value calculation using Ground Truth image
	long float rms = 0, count = 0, max_disp = 0;
	for (int i = 0; i < gt.size().width; i++) {
		for (int j = 0; j < gt.size().height; j++) {
			if (disparity.at<uchar>(j, i) != 255) // disparity is not white 255
			{
				int a = abs(disparity.at<uchar>(j, i) - gt.at<uchar>(j, i) / disp_ratio);
				if (a > max_disp)
					max_disp = a;
				rms = rms + a*a; // calculate RMS 
				count = count + 1;
			}
		}
	}
	printf("\n No. of disparities--->%d,   rms(SQRT(sum(diff(disparity-ground truth))^2)/number of disparities)--->%f", (int)count, sqrt(rms / count)); // SQRT RMS and divide by number of disparities

	//Normalization to Gray scale
	for (int i = 0; i < gt.size().width; i++)
		for (int j = 0; j < gt.size().height; j++)
			if (disparity.at<uchar>(j, i) != 255)
				disparity.at<uchar>(j, i) = disparity.at<uchar>(j, i) * disp_ratio * 255 / max_disp;

	finish_time = getTickCount(); //Stop the processing time
	printf("\nElapsed Time : %.2lf sec \n", (finish_time - start_time) / getTickFrequency());         //print the processing time in seconds
	cout << "----------------------------------------------------------------------------" << endl;
	imshow("disparity_" + alg + ";    rms:" + to_string(sqrt(rms / count)) /*+ "    bp%: " + to_string(bad[0] / count)*/, disparity);
	imwrite("disparity_" + alg + "_" + disparity_type + "_RMS.png", disparity);
	waitKey();
	destroyAllWindows();
}

/**
 * functions that takes the images matrixes and calculates LOGOS matches.
 * 
 *
 *
 * @param input two images matrixes
 * @return VOID, shows the LOGOS matches.
 */
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
	//namedWindow("output", WINDOW_NORMAL);
	//resizeWindow("output", output.cols / SCALE, output.rows / SCALE);
	imshow("testing LOGOS output", output);
	waitKey(0);
	imwrite("testing LOGOS output.png", output);
	destroyAllWindows();
}

/**
 * functions that sorts the areas in descending order. Helper method for the sort method used
 * in createPortraitMode function to sort contours
 *
 * @param input two contours
 * @return BOOL, return which contour is bigger in area.
 */
bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
	double i = fabs(contourArea(cv::Mat(contour1)));
	double j = fabs(contourArea(cv::Mat(contour2)));
	return (i > j);
}

/**
 * functions that creates a portrait mode out of two stereo images pair. 
 *
 * @param input two images matrixes
 * @return VOID, The resultant image will be shown on screen using imshow.
 */
void createPortraitMode(Mat img_1, Mat img_2)
{
	unsigned long start_time = 0, finish_time = 0; 
	start_time = getTickCount(); //start processing time
	Ptr<Feature2D> f2d;
	Ptr<DescriptorMatcher> matcher;
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;

	f2d = SIFT::create();
	matcher = new FlannBasedMatcher(new flann::KDTreeIndexParams(4));

	for (int i = 0; i < img_1.cols; i++)
		for (int j = 0; j < img_1.rows; j++) //Start Dense disparity exactly how it is done in the previous function
		{
			keypoints_1.push_back(KeyPoint(i, j, 1));
			keypoints_2.push_back(KeyPoint(i, j, 1));
		}
	f2d->compute(img_1, keypoints_1, descriptors_1); // To get descriptors at each and every pixel(dense disparity)
	f2d->compute(img_2, keypoints_2, descriptors_2); // To get descriptors at each and every pixel(dense disparity)

	vector< DMatch > matches;
	matcher->match(descriptors_1, descriptors_2, matches); // matching 

	vector<DMatch> matchesGMS;
	cv::xfeatures2d::matchGMS(img_1.size(), img_2.size(), keypoints_1, keypoints_2, matches, matchesGMS); // apply GMS to eliminate outliers
	matches = matchesGMS;
	//Calculating dispairity and disparity map:
	Mat disparity(img_1.size().height, img_1.size().width, CV_8U, Scalar(255));
	for (int i = 0; i < (int)matches.size(); i++)
	{
		int x = keypoints_1[matches[i].queryIdx].pt.x; // good matches index of img1 descriptor, after the index substitute into the keypoints of img1 then get the x-coordinates of the keypoint
		int y = keypoints_1[matches[i].queryIdx].pt.y; // good matches index of img1 descriptor, after the index substitute into the keypoints of img1 then get the y-coordinates of the keypoint
		int x1 = keypoints_2[matches[i].trainIdx].pt.x; // good matches index of img2 descriptor, after the index substitute into the keypoints of img1 then get the x - coordinates of the keypoint
		disparity.at<uchar>(y, x) = abs(x - x1); // calculate disparity
	}
	namedWindow("Disparity Map - All matches_GMS", WINDOW_NORMAL);
	cv::resizeWindow("Disparity Map - All matches_GMS", disparity.cols / 3, disparity.rows / 3);
	imshow("Disparity Map - All matches_GMS", disparity);
	waitKey();
	imwrite("Disparity Map - All matches_GMS.png", disparity);
	destroyAllWindows();

	for (int i = 0; i < disparity.cols; i++) {
		for (int j = 0; j < disparity.rows; j++) {
			if (disparity.at<uchar>(j, i) == 255) {
				disparity.at<uchar>(j, i) = 0; // for contrast purpuoses, replace white pixels with black pixels
			}
		}
	}
	namedWindow("Disparity Map - All matches_GMS_After_for_loop", WINDOW_NORMAL);
	cv::resizeWindow("Disparity Map - All matches_GMS_After_for_loop", disparity.cols / 3, disparity.rows / 3);
	imshow("Disparity Map - All matches_GMS_After_for_loop", disparity);
	waitKey();
	imwrite("Disparity Map - All matches_GMS_After_for_loop.png", disparity);
	destroyAllWindows();
	
	Mat kernel;
	Point anchor;
	double delta = 0;
	int ddepth = -1;	
	int kernel_size;
	kernel_size = 9;
	Mat thresh;
	Mat img_bw;
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size); // i created this kernel in case i needed to apply futher smoothing
	Mat blurred;
	int threshhold = 60;// the threshold value, i have tested different values and 60 turns out to be a good one 
	threshold(disparity, thresh, threshhold, 255, THRESH_BINARY); //thresholding of pixels to white
	
	namedWindow("Right after thresholding", WINDOW_NORMAL);
	cv::resizeWindow("Right after thresholding", thresh.cols / 3, thresh.rows / 3);
	imshow("Right after thresholding", thresh);
	waitKey();
	imwrite("Right after thresholding.png", thresh);
	
	Mat img_final;
	dilate(thresh, img_final, Mat(), Point(-1, -1), 2, 1, 1); // dilation function to get rid of the noise and get the image object blob
	
	namedWindow("Dialation", WINDOW_NORMAL);
	cv::resizeWindow("Dialation", img_final.cols / 3, img_final.rows / 3);
	imshow("Dialation", img_final);
	waitKey();
	imwrite("Dialation.png", img_final);

	// detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img_final, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);
	// draw contours on the original image
	drawContours(img_final, contours, -1, Scalar(0, 255, 0), 2);
	
	namedWindow("contours", WINDOW_NORMAL);
	cv::resizeWindow("contours", img_final.cols / 3, img_final.rows / 3);
	imshow("contours",img_final);
	waitKey(0);
	destroyAllWindows();
	imwrite("contours.png",img_final);
	
	int largest_contour_index = 0;
	int largest_area = 0;
	Rect bounding_rect;

	std::sort(contours.begin(), contours.end(), compareContourAreas); // sort contours in descending order using helper function

	//Draw the contour and rectangle
	int numContours = 5; // number is not supposed to be hard coded but function should be enhanced to find the needed number of contours
	for (int i = 0; i < numContours; i++) {
		drawContours(img_final, contours, i, Scalar(100, 255, 255), FILLED, 8, hierarchy); // draw the contours of the five biggest contours 
	}
	
	rectangle(img_final, bounding_rect, Scalar(0, 255, 0), 2, 8, 0);
	namedWindow("Display Contours", WINDOW_NORMAL);
	cv::resizeWindow("Display Contours", img_final.cols / 3, img_final.rows / 3);
	imshow("Display Contours", img_final);
	waitKey(0);
	destroyAllWindows();
	imwrite("Display Contours.png", img_final);

	Mat image_blurred;
	medianBlur(img_1, image_blurred, 15); // blur the original image
	namedWindow("Blurred imaged before switching pixels", WINDOW_NORMAL);
	cv::resizeWindow("Blurred imaged before switching pixels", image_blurred.cols / 3, image_blurred.rows / 3);
	imshow("Blurred imaged before switching pixels", image_blurred);
	waitKey();
	destroyAllWindows();
	imwrite("Blurred imaged before switching pixels.png", image_blurred);
	
	for (int i = 0; i < thresh.rows - 3; i++) {
		for (int j = 0; j < thresh.cols - 3; j++) {
			if (img_final.at<uchar>(i, j) != 255 && img_final.at<uchar>(i, j) != 0) { // find the grey pixels inside the contours
				// replace the grey pixel with the original pixels so the object will
				// return back unblurred while all other pixels will be blurred to get 
				// the needed portrait mode
				image_blurred.at<cv::Vec3b>(i, j)[0] = img_1.at<cv::Vec3b>(i, j)[0]; 
				image_blurred.at<cv::Vec3b>(i, j)[1] = img_1.at<cv::Vec3b>(i, j)[1];
				image_blurred.at<cv::Vec3b>(i, j)[2] = img_1.at<cv::Vec3b>(i, j)[2];
			}
		}
	}
	namedWindow("Blurred imaged based on depth", WINDOW_NORMAL);
	cv::resizeWindow("Blurred imaged based on depth", image_blurred.cols / 3, image_blurred.rows / 3);
	imshow("Blurred imaged based on depth", image_blurred);
	
	namedWindow("Original Image", WINDOW_NORMAL);
	cv::resizeWindow("Original Image", img_1.cols / 3, img_1.rows / 3);
	imshow("Original Image", img_1);
	waitKey();
	destroyAllWindows();
	imwrite("Blurred imaged based on depth.png", image_blurred);

	finish_time = getTickCount(); //stop processing time
	printf("Elapsed Time : %.2lf sec \n", (finish_time - start_time) / getTickFrequency());         //check processing time  
	cout << "-------------------------------------------------------------------------------" << endl;
}

int main(int argc, char* argv[])
{
	
	cout << "########################### Sparse Disparity ###################################" << endl;
	int disp_ratio = 4; // depends on the dataset ground truth scaling factor
	String disparity_type = "sparse"; // set to sparse
	Mat img1 = imread("../SourceImages/left1.png");
	Mat img2 = imread("../SourceImages/right1.png");
	Mat gt = imread("../SourceImages/left_gt1.png"); // Ground Truth image
	imshow("Original Left Image ",img1);
	imshow("Original Right Image ", img2);
	imshow("Original Ground Truth Image ", gt);
	cout << "Press any key to continue" << endl;
	waitKey();
	matchBasedDispCalculate(img1, img2, gt, "sift", disp_ratio, disparity_type);
	matchBasedDispCalculate(img1, img2, gt, "orb", disp_ratio, disparity_type);
	matchBasedDispCalculate(img1, img2, gt, "LOGOS", disp_ratio, disparity_type);
	matchBasedDispCalculate(img1, img2, gt, "GMS", disp_ratio, disparity_type);
	cout << "Press any key to continue" << endl;
	waitKey();

	cout << "########################### Dense Disparity ###################################" << endl;
	imshow("Original Left Image ", img1);
	imshow("Original Right Image ", img2);
	imshow("Original Ground Truth Image ", gt);
	cout << "Press any key to continue" << endl;
	waitKey();
	disparity_type = "dense"; // set to dense
	matchBasedDispCalculate(img1, img2, gt, "sift", disp_ratio, disparity_type);
	matchBasedDispCalculate(img1, img2, gt, "orb", disp_ratio, disparity_type);
	matchBasedDispCalculate(img1, img2, gt, "GMS", disp_ratio, disparity_type);
	cout << "Press any key to continue" << endl;
	waitKey();
	
	cout << "########################### Panorama Mode ###################################" << endl;
	Mat img3 = imread("../SourceImages/leftRobot.png");
	Mat img4 = imread("../SourceImages/rightRobot.png");
	namedWindow("Original Left Image", WINDOW_NORMAL);
	cv::resizeWindow("Original Left Image", img3.cols / 3, img3.rows / 3);
	imshow("Original Left Image", img3);

	namedWindow("Original Right Image", WINDOW_NORMAL);
	cv::resizeWindow("Original Right Image", img4.cols / 3, img4.rows / 3);
	imshow("Original Right Image", img4);
	cout << "Press any key to continue" << endl;
	waitKey();
	createPortraitMode(img3, img4);
	
	Mat disp3; 
	stereo_match(img3, img4, disp3); // test BM disparity
	namedWindow("Disparity Map using Block Matching", WINDOW_NORMAL);
	imshow("Disparity Map using Block Matching", disp3);
	imwrite("Disparity Map using Block Matching.png", disp3);
	waitKey(0);
	return 0;
}

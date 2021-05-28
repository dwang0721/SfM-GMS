#include "CalibrationUtil.h"

int addChessboardPoints(const vector<string>& imgList, Size& boardSize, vector<vector<Point3f>>&  objPtr, vector<vector<Point2f>>& imgPtr, bool drawCorner){
    
    // points on the chessboard
    vector<Point2f> imageCorners;
    vector<Point3f> objectCorners;
    Mat image;
    int successes = 0;

    cout << "\nStart Camera Calibration" << endl;
    // initialize the chessboard corners
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {

            objectCorners.push_back(Point3f(i, j, 0.0f));
        }
    }

    // go through all chessboard images
    for (int i=0; i<imgList.size(); i++){
        Mat greyImage= imread(imgList[i],0);

        // get the chessboard corners
        cout << "looking for chessboard Corners: " << imgList[i] << endl;
        bool found = findChessboardCorners(greyImage, boardSize, imageCorners);

        if (!found){
            cout << "Chessboard detection failed: " << imgList[i] << endl;
            continue;
        }

        cout << "Chessboard detected..." << imgList[i] << endl;
        // define criteria for sub pixel accuracy
        TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 30, 0.1);
        //refine corner location based on criteria.
        cornerSubPix(greyImage, imageCorners, Size(5,5), Size(-1,-1), criteria);

        if(imageCorners.size() == boardSize.area()){
            imgPtr.push_back(imageCorners); 
            objPtr.push_back(objectCorners);
            successes +=1 ;
        }

        // draw the Corners
        //drawChessboardCorners(greyImage, boardSize, imageCorners, found);
        //namedWindow("Corners on Chessboard", WINDOW_NORMAL);
        //resizeWindow("Corners on Chessboard", greyImage.cols/4, greyImage.rows/4);
        //imshow("Corners on Chessboard", greyImage);
        //waitKey(0);
    }   
    return successes;
}
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/core/cvstd.hpp"
#include <iostream>
#include <cstdlib>
#include <stdio.h>


/** @function main */
int main(int argc, char** argv)
{


	cv::Size patternsize(6, 5); ///interior number of corners

	/// Load the images
	cv::String folder = "data/checkerboard_images/*.png"; ///path to the images
	std::vector<cv::String> filenames;
	cv::glob(folder, filenames);
	cv::Point3f point3(.0f, .0f, .0f);
	cv::Point2f point2(.0f, .0f);
	std::vector<std::vector<cv::Point3f>> objectPoints( filenames.size(), std::vector<cv::Point3f>(30, point3) );
	std::vector<std::vector<cv::Point2f>> imagePoints( filenames.size(), std::vector<cv::Point2f>(30, point2) );
	cv::Size imageSize;
	///create vector of 3D coordinates 57*30*3
	for (int i = 0; i < filenames.size(); ++i) {
		for (int v = 0; v < 30; ++v) {
			cv::Point3f coord(v%6, int(v/6), 0);
			objectPoints[i][v] = coord;
			//std::cout << objectPoints[i][v]<<" ";
		}
		//std::cout << "\n";
	}

	///find coordinates for the corners
	for (int i = 0; i < filenames.size(); ++i) {
		std::cout <<"processing " <<filenames[i] << "\r";

		cv::Mat src = cv::imread(filenames[i], 0);

		imageSize = cv::Size(src.rows, src.cols);

		if (src.empty()) {
			std::cout << "Image not found\n";
			return -1;
		}

		std::vector<cv::Point2f> corners; ///this will be filled by the detected corners

		///CALIB_CB_FAST_CHECK saves a lot of time on images
		///that do not contain any chessboard corners
		bool patternfound = cv::findChessboardCorners(src, patternsize, corners,
														cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
														+ cv::CALIB_CB_FAST_CHECK);

		if (patternfound) cv::cornerSubPix(src, corners, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));


		for (int c = 0;c < corners.size();++c) {
			imagePoints[i][c] = corners[c];
		}

		//cv::drawChessboardCorners(src, patternsize, corners, patternfound);
		//cv::namedWindow("source_window");
		//cv::imshow("source_window", src);
		//cv::waitKey(0);
	}


	cv::Mat cameraMatrix, distCoeffs;
	std::vector<cv::Mat>  R, T;

	std::cout << "\ncalibrating..."<< std::endl;

	float re_progectionError = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, R, T);
	/*
	std::cout << "re_progectionError : " << re_progectionError << std::endl;
	std::cout << "cameraMatrix : \n" << cameraMatrix << std::endl;
	std::cout << "distCoeffs : \n" << distCoeffs << std::endl;
	std::cout << "Rotation vector : \n" << R << std::endl;
	std::cout << "Translation vector : \n" << T << std::endl;
	*/
	std::vector<cv::Point2f> estImagePoints;
	//retro error calculation per image
	double max = 0, min = 10000;
	int argmax, argmin;
	for (int i = 0; i < objectPoints.size(); ++i) {
		cv::projectPoints(objectPoints[i], R[i], T[i], cameraMatrix, distCoeffs, estImagePoints);// NOT WORKING
		double sum = 0;
		for (int j = 0;j < 30;++j) {
			sum += cv::norm(cv::Mat(estImagePoints[j]), cv::Mat(imagePoints[i][j]), cv::NORM_L2);
			
		}
		if (sum > max) { max = sum; argmin = i; }
		if (sum < min) { min = sum; argmax = i; }
	}
	std::cout << "min: " << filenames[argmin] << std::endl;
	std::cout << "max: " << filenames[argmax] << std::endl;

	cv::Mat newCameraMatrix, map1, map2, rect;
	int m1type = CV_32FC1;
	
	cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, rect, newCameraMatrix, imageSize, m1type, map1, map2);

	cv::Mat src = cv::imread("data/test_image.png", 1);
	cv::Mat dst = cv::Mat::zeros(map1.rows, map1.cols, src.type());
	int interpolation= cv::INTER_LINEAR, borderMode = cv::BORDER_CONSTANT;
	const cv::Scalar& borderValue = cv::Scalar();

	cv::remap(src, dst, map1, map2, interpolation, borderMode, borderValue);

	double fy = 0, fx = 0;
	cv::Mat dst2, dst1;
	cv::Size dsize(imageSize.height/4, imageSize.width/3);

	cv::resize(dst, dst2, dsize, fx, fy, interpolation);
	cv::resize(src, dst1, dsize, fx, fy, interpolation);

	cv::namedWindow("source_window");
	cv::imshow("source_window", dst1);

	cv::namedWindow("dest_window");
	cv::imshow("dest_window", dst2);

	cv::waitKey(0);

	return 0;
}


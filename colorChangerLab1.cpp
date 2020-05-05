#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cstdlib>
#include <stdio.h>

using namespace cv;
using namespace std;

void changeShirtsHSV(Mat* img, Vec3b mean) {
	Mat hsv = *img;
	Vec3b pixel(0, 0, 0);
	int count = 0;
	Mat rgb = Mat::zeros(hsv.rows, hsv.cols, hsv.type());

	for (int y = 0; y < hsv.rows; ++y) {
		for (int x = 0; x < hsv.cols; ++x) {
			pixel = hsv.at<Vec3b>(y, x);
			if (abs(pixel[0] - mean[0])< 8 && pixel[1]>50 && pixel[2]>50) {
				hsv.at<Vec3b>(y, x)[0] += 30;
				++count;
			}
		}
	}
	cvtColor(hsv, rgb, COLOR_HSV2BGR);

	imshow("source_window", rgb);
	cout << count << endl;
}

void changeShirts(Mat* img, Vec3b mean){

	Vec3b pixel(0, 0, 0);
	Vec3b color(201, 37, 92);
	Mat src = *img;


	for (int y = 0; y < src.rows; ++y) {
		for (int x = 0; x < src.cols; ++x) {
			pixel = src.at<Vec3b>(y, x);
			if ( sqrt(pow(pixel[0]-mean[0],2)+ pow(pixel[1]-mean[1],2)+ pow(pixel[2]- mean[2],2)) < 50) {
				src.at<Vec3b>(y, x) = color;
			}
		}
	}

	imshow("source_window", src);
}

void onMouse(int event, int x, int y, int flags, void* userdata)
{
	Mat img = *(Mat*)userdata;

	/// TO DO check for pixels near img borders 
	if (event == EVENT_LBUTTONDOWN) {
		
		Vec3f vmean = Vec3f(0, 0, 0);
		float hueMean = 0;
		Mat hsv = Mat::zeros(img.rows, img.cols, img.type());
		/// convert img to HSV in hsv
		cvtColor(img, hsv, COLOR_BGR2HSV);

		for (int i = y - 4; i <= y + 4; ++i) {
			for (int j = x - 4; j <= x + 4; ++j) {
				vmean += img.at<Vec3b>(i, j);
				hueMean += hsv.at<Vec3b>(i, j)[0];
			}
		}
		hueMean = hueMean / 81;
		vmean = vmean / 81;

		cout << "hueMean: " << hueMean<< endl;

		///changeShirts(&img, vmean);
		changeShirtsHSV(&hsv, hueMean);
	}
	
}

/** @function main */
int main(int argc, char** argv)
{
	/// Load the image
	Mat src = imread(argv[1], 1);
	if (src.empty()) {
		cout << "Image not found" << endl;
		return -1;
	}

	/// Show what you got
	namedWindow("source_window");
	imshow("source_window", src);

	setMouseCallback("source_window", onMouse, (void*) &src);

	/// Wait until user exits the program
	waitKey(0);

	return 0;
}


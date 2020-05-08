#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

/// Global variables

cv::Mat src, src_gray, smoothed, canny_edges, detected_edges, edges_gray, temp;
int circle_threshold = 19;
int edgeThresh = 1;
int thresholdCanny = 425, thresholdHough=130;
int const max_lowThreshold = 600;
int ratio = 2;
int kernel_size = 3;
std::string window_name = "Edge Map";

void CannyThreshold(int, void*) {
	cv::Canny(smoothed, detected_edges, thresholdCanny, thresholdCanny*ratio, kernel_size);
	canny_edges = cv::Scalar::all(0);

	src.copyTo(canny_edges, detected_edges);
	imshow(window_name, canny_edges);
}


void houghThreshold(int, void*) {
	temp = canny_edges.clone();
	// Standard Hough Line Transform
	std::vector<cv::Vec2f> lines; // will hold the results of the detection
	cv::HoughLines(edges_gray, lines, 1, CV_PI / 60, thresholdHough); // runs the actual detection
	// Draw the lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(temp, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
		cv::imshow("hough_edges", temp);
	}
}

void HoughCircles(int, void*) {
	temp = canny_edges.clone();
	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(edges_gray, circles, cv::HOUGH_GRADIENT, 1, 1, thresholdCanny*ratio, circle_threshold, 0, 0);

	for (size_t i = 0; i < circles.size(); i++)
	{
		std::cout << circles[i];
		cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// draw the circle center
		cv::circle(temp, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
		// draw the circle outline
		cv::circle(temp, center, radius, cv::Scalar(0, 255, 0), 3, 8, 0);
	}
	std::cout << "\n";
	cv::imshow("circles", temp);
}
int main(int argc, char** argv)
{
	src = cv::imread(argv[1]);
	if (!src.data)
	{
		return -1;
	}

	canny_edges.create(src.size(), src.type());
	cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
	smoothed = src_gray;
	
	cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
	cv::createTrackbar("Min Threshold:", window_name, &thresholdCanny, max_lowThreshold, CannyThreshold);
	CannyThreshold(0, 0);
	cv::waitKey(0);
	cv::destroyAllWindows();

	cv::cvtColor(canny_edges, edges_gray, cv::COLOR_BGR2GRAY);
	cv::namedWindow("hough_edges", cv::WINDOW_AUTOSIZE);
	cv::createTrackbar("Min Threshold:", "hough_edges", &thresholdHough, 300, houghThreshold);
	houghThreshold(0, 0);
	cv::waitKey(0);
	cv::destroyAllWindows();

	//cv::blur(edges_gray, smoothed, cv::Size(3, 3));
	cv::namedWindow("circles", cv::WINDOW_AUTOSIZE);
	cv::createTrackbar("Min Threshold:", "circles", &circle_threshold, 100, HoughCircles);
	HoughCircles(0,0);
	cv::waitKey(0);

	return 0;
}
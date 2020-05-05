#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/core/cvstd.hpp"
#include "filter.h"
#include <iostream>
#include <cstdlib>
#include <stdio.h>


// hists = vector of 3 cv::mat of size nbins=256 with the 3 histograms
// e.g.: hists[0] = cv:mat of size 256 with the red histogram
//       hists[1] = cv:mat of size 256 with the green histogram
//       hists[2] = cv:mat of size 256 with the blue histogram
void showHistogram(std::vector<cv::Mat>& hists)
{
	// Min/Max computation
	double hmax[3] = { 0,0,0 };
	double min;
	cv::minMaxLoc(hists[0], &min, &hmax[0]);
	cv::minMaxLoc(hists[1], &min, &hmax[1]);
	cv::minMaxLoc(hists[2], &min, &hmax[2]);

	std::string wname[3] = { "blue", "green", "red" };
	cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
							 cv::Scalar(0,0,255) };

	std::vector<cv::Mat> canvas(hists.size());

	// Display each histogram in a canvas
	for (int i = 0, end = hists.size(); i < end; i++)
	{
		canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < hists[0].rows - 1; j++)
		{
			cv::line(
				canvas[i],
				cv::Point(j, rows),
				cv::Point(j, rows - (hists[i].at<float>(j) * rows / hmax[i])),
				hists.size() == 1 ? cv::Scalar(200, 200, 200) : colors[i],
				1, 8, 0
			);
		}

		cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
	}
}


int histogramEqualization()
{
	cv::Mat src = cv::imread("data/countryside.jpg",1);
	if (src.empty()) {
		std::cout << "Image not found" << std::endl;
		return -1;
	}
	std::vector<cv::Mat> cha;
	
	cv::split(src,cha);

	int bins = 256;
	int histSize[] = { bins };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };
	cv::Mat equs[3];
	std::vector<cv::Mat> hists;
	
	int channels[] = { 0 };
	for (int i = 0;i < 3;++i) {
		cv::MatND hist;
		calcHist(&cha[i], 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
		hists.push_back(hist);
		equalizeHist(cha[i], equs[i]);
	}
	showHistogram(hists);
	
	cv::Mat merged;
	cv::namedWindow("source_window");
	cv::merge(equs, 3, merged);
	cv::imshow("source_window", merged);
	cv::waitKey();
	//----------------------------------
	for (int hsvChannel = 0;hsvChannel < 3; ++hsvChannel) {

		cv::Mat hsv;
		cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

		std::vector<cv::Mat> cha2;
		cv::split(hsv, cha2);
		cv::Mat equs1[3];
		equalizeHist(cha2[hsvChannel], cha2[hsvChannel]);

		std::vector<cv::Mat> hists2;

		for (int i = 0;i < 3;++i) {
			equs1[i] = cha2[i];
			cv::MatND hist2;
			calcHist(&cha2[i], 1, channels, cv::Mat(), hist2, 1, histSize, ranges, true, false);
			hists2.push_back(hist2);
			equalizeHist(cha[i], equs[i]);
		}
		showHistogram(hists2);

		cv::Mat merged1;
		cv::merge(equs1, 3, merged1);
		cv::cvtColor(merged1, hsv, cv::COLOR_HSV2BGR);
		cv::namedWindow("source_image");
		cv::imshow("source_image", src);
		cv::waitKey();
		cv::namedWindow("equalized_image");
		cv::imshow("equalized_image", hsv);
		cv::waitKey();
		cv::destroyAllWindows();
	}
	return 0;
}

void on_trackbar_kernel_median(int pos, void* data) {
	cv::Mat src = *(cv::Mat*)data, dst;
	if (pos % 2 == 0) pos++;
	cv::medianBlur(src, dst, pos);
	cv::imshow("median", dst);
}
/*
void on_trackbar_kernel_gauss(int pos, void* data) {
	cv::Mat src = *(cv::Mat*)data, dst;
	if (pos % 2 == 0) pos++;
	cv::GaussianBlur(src, dst, pos, );
	cv::imshow(window_name, dst);

void on_trackbar_variance_gauss(int pos, void* data) {
	cv::Mat src = *(cv::Mat*)data, dst;
	if (pos % 2 == 0) pos++;
	cv::GaussianBlur(src, dst, pos);
	cv::imshow("median", dst);
}

*/
int imageFiltering() {
	cv::Mat src = cv::imread("data/image.jpg", 1);
	cv::resize(src, src, cv::Size (0, 0), 0.5, 0.5, cv::INTER_LINEAR);
	if (src.empty()) {
		std::cout << "Image not found" << std::endl;
		return -1;
	}

	int ksize = 1;
	Filter m_filter(src,1);
	//GaussianFilter g_filter(src);
	//BilateralFilter b_filter(src,1,1,1);
	cv::namedWindow("median");
	cv::imshow("median", src);
	cv::createTrackbar("kernel size", "median", &ksize, 29, on_trackbar_kernel_median, (void*) &src);
	cv::waitKey();
	/*cv::namedWindow("gaussian");
	cv::imshow("gaussian", src);
	cv::createTrackbar("kernel size", "gaussian", &ksize, 29, on_trackbar_kernel_gauss, (void*)&src);
	cv::createTrackbar("sigma", "gaussian", &ksize, 29, on_trackbar_variance_gauss, (void*)&src);
	cv::waitKey();
	cv::namedWindow("bilateral");
	cv::imshow("bilateral", src);
	cv::createTrackbar("sigma_range", "bilateral", &ksize, 29, on_trackbar_kernel_gauss, (void*)&src);
	cv::createTrackbar("sigma_space", "bilateral", &ksize, 29, on_trackbar_variance_gauss, (void*)&src);
	*/
}


int main(int argc, char** argv) {
	//histogramEqualization();
	imageFiltering();
	return 0;
}
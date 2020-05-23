#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "panoramic_utils.h"
#include <iostream>
#include <cstdlib>
#include <stdio.h>

int main(int argc, char** argv)
{
	// get video
	cv::String filename = "data/Lab_6_data/video.mov";
	auto cap = cv::VideoCapture(filename);

	// get objects
	cv::String folder = "data/Lab_6_data/objects/*.png"; //images folder
	std::vector<cv::String> filenames;
	std::vector<cv::Mat> objects; // books
	cv::Mat obj, big_obj, frame, matches_image;
	cv::glob(folder, filenames);

	//KeyPoints, descriptors and matches vectors
	std::vector<cv::KeyPoint> frm_key_points;
	std::vector<std::vector<cv::KeyPoint>> objs_key_points;
	
	cv::Mat frm_descriptors;
	std::vector<cv::Mat> objs_descriptors;
	
	std::vector<cv::DMatch> matches, best_matches; 

	//default parameters still need to try changing them (tried only a little)
	auto ORB = cv::ORB::create(1000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

	auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);


	//load objects , resize and turn to grayscale
	for (int i = 0; i < filenames.size(); ++i) {
		obj = cv::imread(filenames[i], 1);

		//center image to avoid missing borders with ORB (finds sometimes book corners)
		cv::Mat centered_obj(obj.rows + 400, obj.cols + 200, obj.type());
		obj.copyTo(centered_obj(cv::Rect(100, 100, obj.cols, obj.rows)));
		objects.push_back(centered_obj);

		/*   RESIZE, EQUALIZE AND BLUR, did not help
		cv::Size dsize(big_obj.cols / 3, big_obj.rows / 3);
		cv::resize(big_obj, obj, dsize, 0, 0, cv::INTER_LINEAR);
		cv::cvtColor(objects[i], objects[i], cv::COLOR_BGR2GRAY);
		cv::equalizeHist(objects[i], objects[i]);
		cv::GaussianBlur(objects[i], objects[i], cv::Size(3, 3), 1, 1);
		*/
	}
	//compute objects descriptors
	ORB->detect(objects, objs_key_points);
	ORB->compute(objects, objs_key_points, objs_descriptors);

	//load first frame
	cap.read(frame);

	/* //frame hist equalization
	cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(frame, frame);
	*/

	//compute first frame descriptors
	ORB->detect(frame, frm_key_points);
	ORB->compute(frame, frm_key_points, frm_descriptors);

	//match descriptors
	matcher->match(objs_descriptors[0], frm_descriptors, matches);

	// find best matches
	float min_dist = INFINITY;
	for (auto match : matches) {
		if (match.distance < min_dist) {
			min_dist = match.distance;
		}
	}
	float dist_threshold = min_dist * 2;
	for (size_t j = 0;j < matches.size();++j) {
		if (matches[j].distance <= dist_threshold) {
			best_matches.push_back(matches[j]);
		}
	}
	// show matches for each book
	for (int i = 0; i < objects.size(); ++i) {

		cv::drawMatches(objects[i], objs_key_points[i], frame, frm_key_points, best_matches, matches_image);

		//resize for visualization, only for low res monitor
		cv::Mat res;
		cv::Size dsize(matches_image.cols / 2, matches_image.rows / 2);
		cv::resize(matches_image, res, dsize, 0, 0, cv::INTER_LINEAR);

		cv::namedWindow("frame", 1);
		cv::imshow("frame", res);
		cv::waitKey();
	}

	/* //iterate over frames, untested
	while(cap.read(frame) != NULL) {
	}
	*/
	return 0;
}

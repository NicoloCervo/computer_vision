#include "lab6.hpp"
#include "ObjTracker.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/video.hpp""
#include <iostream>
#include <stdio.h>
#include <cstdlib>


cv::Point2f operator*(cv::Mat M, const cv::Point2f& p){

	cv::Mat_<double> src(3, 1);

	src(0, 0) = p.x;
	src(1, 0) = p.y;
	src(2, 0) = 1.0;

	cv::Mat_<double> dst = M * src; 
	return cv::Point2f(dst(0, 0), dst(1, 0));
}

void findMatches(std::vector<cv::Mat> objects, cv::Mat frame, 
					std::vector<std::vector<cv::Point2f>>& obj_points_hom, 
						std::vector<std::vector<cv::Point2f>>& frm_points_hom ) {

	//KeyPoints, descriptors and matches vectors
	std::vector<cv::KeyPoint> frm_key_points;
	std::vector<std::vector<cv::KeyPoint>> objs_key_points;

	cv::Mat frm_descriptors;
	std::vector<cv::Mat> objs_descriptors;

	std::vector<cv::DMatch> matches;
	std::vector<std::vector<cv::DMatch>> best_matches;


	best_matches.resize(objects.size());
			
	auto SIFT = cv::xfeatures2d::SIFT::create(10000);
		
	//compute objects descriptors
	SIFT->detect(objects, objs_key_points);
	SIFT->compute(objects, objs_key_points, objs_descriptors);

	//compute first frame descriptors
	SIFT->detect(frame, frm_key_points);
	SIFT->compute(frame, frm_key_points, frm_descriptors);


	obj_points_hom.resize(objects.size());
	frm_points_hom.resize(objects.size());

	// find matches
	auto matcher = cv::BFMatcher::create(cv::NORM_L2);
	for (int i = 0; i < objects.size(); ++i) {

		//match descriptors
		matcher->match(objs_descriptors[i], frm_descriptors, matches);

		// find best matches
		float min_dist = INFINITY;
		for (auto match : matches) {
			if (match.distance < min_dist) {
				min_dist = match.distance;
			}
		}
		float dist_threshold = min_dist * 4;
		for (size_t j = 0;j < matches.size();++j) {
			if (matches[j].distance <= dist_threshold) {
				best_matches[i].push_back(matches[j]);
			}
		}

		matches.clear();

		for (auto match : best_matches[i]) {
			obj_points_hom[i].emplace_back(objs_key_points[i][match.queryIdx].pt);
			frm_points_hom[i].emplace_back(frm_key_points[match.trainIdx].pt);
		}

	}
}

int main(int argc, char** argv)
{
	// get video
	cv::String filename = "C:/Lab6/Lab_6_data/video.mov";
	auto cap = cv::VideoCapture(filename);

	// get objects file names
	cv::String folder = "C:/Lab6/Lab_6_data/objects/*.png"; //images folder
	std::vector<cv::String> filenames;
	std::vector<cv::Mat> objects;
	cv::glob(folder, filenames);
	
	cv::Mat obj, frame, frame_temp;
	std::vector < std::vector<cv::Point2f>> obj_points_hom, frm_points_hom;
	std::vector<cv::Point2f> proj_corners;
	
	//load first frame
	cap.read(frame);

	//load objects
	for (int i = 0; i < filenames.size(); ++i) {
		obj = cv::imread(filenames[i], 1);
		objects.push_back(obj);
	}
		
	findMatches(objects, frame, obj_points_hom, frm_points_hom);

	/*---------------------------from here your code started--------------------------------*/

	for (int i = 0; i < objects.size(); ++i) {

		cv::Mat mask;

		cv::Mat H = cv::findHomography(obj_points_hom[i], frm_points_hom[i], mask, cv::RANSAC);

		std::vector<cv::Point2f> corners{
			cv::Point2f(0, 0),
			cv::Point2f(0, objects[i].rows - 1),
			cv::Point2f(objects[i].cols - 1, objects[i].rows - 1),
			cv::Point2f(objects[i].cols - 1,0)
		};

		for (int j = 0; j < 4; ++j) {
			proj_corners.push_back(H*corners[j]);
		}
		frame_temp = frame.clone();
		for (int j = 0; j < objects.size(); ++j) {
			cv::line(frame_temp, proj_corners[j], proj_corners[(j+1)% objects.size()], cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
		}

		corners.clear();
		proj_corners.clear();
	}


	std::vector<std::vector<cv::Point2f>> src_points = frm_points_hom;;
	std::vector<cv::Point2f> dest_points;
	std::vector<uchar> status;
	std::vector<float> err;
	std::vector<cv::KeyPoint> key_points;///
	cv::Mat next_frame, mask, H;
	cv::Size win{ 21, 21 };
	//iterate over frames

	while (cap.read(next_frame)) {
		for (int i = 0; i < objects.size(); i++) {

			cv::calcOpticalFlowPyrLK(frame, next_frame, src_points[i], dest_points, status, err, win);
			H = cv::findHomography(src_points[i], dest_points, mask, cv::RANSAC);
			cv::KeyPoint::convert(dest_points, key_points);
			cv::drawKeypoints(next_frame, key_points, frame_temp);
			cv::imshow("frame", frame_temp);
			cv::waitKey(1);
			src_points[i] = dest_points;
			dest_points.clear();
			key_points.clear();
		}

		frame = next_frame;
	}

	cv::waitKey();
	return 0;
}

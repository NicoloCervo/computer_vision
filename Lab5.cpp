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
	// get images paths
	cv::String folder = "data/datasetL5/*.png"; //images folder
	std::vector<cv::String> filenames;
	cv::glob(folder, filenames);

	int count;
	std::vector<int> shifts;
	std::vector<cv::Mat> images, cyl_projs, descriptors;
	std::vector<std::vector<cv::KeyPoint>> key_points;
	std::vector<std::vector<cv::DMatch>> matches, close_matches;

	double ratio = 4, min_dist, dist_threshold;

	cv::Mat img_keypoints, matches_img, resized_pan;

	for (int i = 0; i < filenames.size(); ++i) {
		images.push_back(cv::imread(filenames[i], 1));
		cyl_projs.push_back(PanoramicUtils::cylindricalProj(images[i], 33));
	}

	auto ORB = cv::ORB::create();

	ORB->detect(cyl_projs, key_points);
	ORB->compute(cyl_projs, key_points, descriptors);

	auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
	matches.resize(descriptors.size() - 1);
	close_matches.resize(descriptors.size() - 1);
	shifts.resize(descriptors.size());

	for (size_t i = 1; i < descriptors.size(); ++i) {
		matcher->match(descriptors[i - 1], descriptors[i], matches[i - 1]);
		min_dist = INFINITY;
		for (auto match : matches[i - 1]) {
			if (match.distance < min_dist) {
				min_dist = match.distance;
			}
		}

		dist_threshold = min_dist * ratio;
		for (size_t j = 0;j < matches[i - 1].size();++j) {
			if (matches[i - 1][j].distance <= dist_threshold) {
				close_matches[i - 1].push_back(matches[i - 1][j]);
			}
		}

		count = 0;
		shifts[0] = 0;
		shifts[i] = 0;
		//calculate the average shift of the keypoints
		for (auto match : close_matches[i - 1]) {
			const cv::KeyPoint kp1 = key_points[i - 1][match.queryIdx], kp2 = key_points[i][match.trainIdx];
			if (abs(kp1.pt.y - kp2.pt.y) < 10) {
				shifts[i] += std::abs(kp1.pt.x - kp2.pt.x);
				++count;
				std::cout << kp1.pt.x - kp2.pt.x << "    ";
			}

		}

		shifts[i] = shifts[i] / count + shifts[i - 1];
		//std::cout << "\n";	std::cout << shifts[i] << "\n";

		cv::drawMatches(cyl_projs[i - 1], key_points[i - 1], cyl_projs[i], key_points[i], close_matches[i - 1], matches_img);
		cv::namedWindow("end_image", cv::WINDOW_AUTOSIZE);
		cv::imshow("end_image", matches_img);
		cv::waitKey(0);
	}

	// Get dimension of final image
	int rows = cyl_projs[0].rows;
	int cols = cyl_projs[0].cols * 13 - shifts[12];

	// Create a black image
	cv::Mat res(rows, cols, cyl_projs[0].type());

	// Copy images in correct position
	for (int i = 0;i < 13;++i) {
		cyl_projs[i].copyTo(res(cv::Rect(shifts[i], 0, cyl_projs[0].cols, rows)));
	}

	//resize image
	cv::Size dsize(cols / 5, rows / 5);
	cv::resize(res, resized_pan, dsize, 0, 0, cv::INTER_LINEAR);
	cv::imshow("end_image", resized_pan);
	cv::waitKey(0);

	return 0;
}

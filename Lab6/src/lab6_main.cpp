#include "lab6.hpp"
#include "ObjTracker.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <iostream>
#include <cstdlib>
#include <stdio.h>


using lab6::Log;

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

    std::vector<cv::DMatch> matches;

    //default parameters still need to try changing them (tried only a little)
    //auto ORB = cv::ORB::create(1000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    auto SIFT = cv::xfeatures2d::SIFT::create(5000);

    //auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    auto matcher = cv::BFMatcher::create(cv::NORM_L2, true);


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
    //ORB->detect(objects, objs_key_points);
    //ORB->compute(objects, objs_key_points, objs_descriptors);
    SIFT->detect(objects, objs_key_points);
    SIFT->compute(objects, objs_key_points, objs_descriptors);

    //load first frame
    cap.read(frame);

    /* //frame hist equalization
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame, frame);
    */

    //compute first frame descriptors
    //ORB->detect(frame, frm_key_points);
    //ORB->compute(frame, frm_key_points, frm_descriptors);
    SIFT->detect(frame, frm_key_points);
    SIFT->compute(frame, frm_key_points, frm_descriptors);

    std::vector<std::vector<cv::DMatch>> best_matches{ objects.size() };

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
        float dist_threshold = min_dist * 6;
        for (size_t j = 0; j < matches.size(); ++j) {
            if (matches[j].distance <= dist_threshold) {
                best_matches[i].push_back(matches[j]);
            }
        }

        cv::drawMatches(objects[i], objs_key_points[i], frame, frm_key_points, best_matches[i], matches_image);

        //resize for visualization, only for low res monitor
        cv::Mat res;
        cv::Size dsize(matches_image.cols / 2, matches_image.rows / 2);
        cv::resize(matches_image, res, dsize, 0, 0, cv::INTER_LINEAR);

        cv::namedWindow("frame", 1);
        cv::imshow("frame", res);
        cv::waitKey();
    }

    /* Prepare object tracking. */
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    std::vector<lab6::ImgObject> templates{ objects.size() };

    for (int i = 0; i < objects.size(); ++i)
    {
        std::vector<cv::Point2f> srcPts;
        srcPts.reserve(best_matches[i].size());
        templates[i].features.reserve(best_matches[i].size());

        for (const auto& match : best_matches[i])
        {
            srcPts.emplace_back(objs_key_points[i][match.queryIdx].pt);
            templates[i].features.emplace_back(frm_key_points[match.trainIdx].pt);
        }

        cv::Mat mask{};
        cv::Mat H{ cv::findHomography(srcPts, templates[i].features, mask, cv::RANSAC) };
        std::vector<cv::Point2f> vertices
        {
            cv::Point2f{ 0, 0 },
            cv::Point2f{ static_cast<float>(objects[i].cols - 1), 0 },
            cv::Point2f{ static_cast<float>(objects[i].cols - 1), static_cast<float>(objects[i].rows - 1) },
            cv::Point2f{ 0, static_cast<float>(objects[i].rows - 1) }
        };
        cv::perspectiveTransform(vertices, templates[i].vertices, H);
    }

    /* Run object tracking. */
    lab6::ObjTracker tracker{ cap, templates };
    if (!tracker.run())
    {
        Log::fatal("Failed to track objects.");
        return -1;
    }

    Log::info("Tracking complete.");
    return 0;
}
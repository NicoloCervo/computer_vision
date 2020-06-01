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

// List of command line arguments.
enum class Arg
{
    vid = 1, // Input video
    obj_folder, // Object templates.
    tot
};

cv::Point2f operator*(cv::Mat M, const cv::Point2f& p) {

    cv::Mat_<double> src(3, 1);

    src(0, 0) = p.x;
    src(1, 0) = p.y;
    src(2, 0) = 1.0;

    cv::Mat_<double> dst = M * src;
    return cv::Point2f(dst(0, 0), dst(1, 0));
}

void findMatches(std::vector<cv::Mat> objects, cv::Mat frame,
    std::vector<std::vector<cv::Point2f>>& obj_points_hom,
    std::vector<std::vector<cv::Point2f>>& frm_points_hom) {

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
        for (size_t j = 0; j < matches.size(); ++j) {
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

using lab6::Log;

int main(int argc, char** argv)
{
    // Command line arguments check.
    if (argc < static_cast<int>(Arg::tot))
    {
        Log::fatal("Required parameters: <video> <templates folder>");
        return -1;
    }

    // get video
    Log::info("Loading video.");
    auto cap = cv::VideoCapture(argv[static_cast<int>(Arg::vid)]);
    if (!cap.isOpened())
    {
        Log::fatal("Failed to open video %s.", argv[static_cast<int>(Arg::vid)]);
        return -1;
    }

    // get objects file names
    std::vector<cv::String> filenames;
    std::vector<cv::Mat> objects;
    cv::glob(argv[static_cast<int>(Arg::obj_folder)], filenames);
    if (filenames.empty())
    {
        Log::fatal("No template images in folder %s.", argv[static_cast<int>(Arg::obj_folder)]);
        return -1;
    }

    cv::Mat obj, frame, frame_temp;
    std::vector < std::vector<cv::Point2f>> obj_points_hom, frm_points_hom;

    //load first frame
    cap.read(frame);

    //load objects
    Log::info("Loading templates.");
    for (int i = 0; i < filenames.size(); ++i) {
        obj = cv::imread(filenames[i], 1);
        if (obj.empty())
        {
            Log::fatal("Failed to open image %s.", filenames[i].c_str());
            return -1;
        }
        objects.push_back(obj);
    }

    Log::info("Computing features.");
    findMatches(objects, frame, obj_points_hom, frm_points_hom);

    /* Prepare object tracking. */
    Log::info("Initialising tracking data.");
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    std::vector<lab6::ImgObject> templates{ objects.size() };

    for (int i = 0; i < objects.size(); ++i)
    {
        templates[i].features.reserve(frm_points_hom[i].size());

        cv::Mat mask;
        cv::Mat H = cv::findHomography(obj_points_hom[i], frm_points_hom[i], mask, cv::RANSAC);

        /*apply mask to the features*/
        for (int mask_idx = 0; mask_idx < mask.rows; mask_idx++) {
            if (mask.at<bool>(mask_idx, 0)) { templates[i].features.emplace_back(frm_points_hom[i][mask_idx]); }
        }

        std::vector<cv::Point2f> vertices
        {
            cv::Point2f{ 0, 0 },
            cv::Point2f{ static_cast<float>(objects[i].cols - 1), 0 },
            cv::Point2f{ static_cast<float>(objects[i].cols - 1), static_cast<float>(objects[i].rows - 1) },
            cv::Point2f{ 0, static_cast<float>(objects[i].rows - 1) }
        };
        cv::perspectiveTransform(vertices, templates[i].vertices, H);

        frame_temp = frame.clone();
        for (int j = 0; j < objects.size(); ++j) {
            cv::line(
                frame_temp,
                templates[i].vertices[j],
                templates[i].vertices[(j + 1) % objects.size()],
                cv::Scalar(255, 255, 255),
                3,
                cv::LINE_AA
            );
        }
    }

    /* Run object tracking. */
    Log::info("Initialising object tracker.");
    lab6::ObjTracker tracker{ cap, templates };
    Log::info("Starting object tracker.");
    if (!tracker.run())
    {
        Log::fatal("Failed to track objects.");
        return -1;
    }

    Log::info("Tracking complete.");
    return 0;
}
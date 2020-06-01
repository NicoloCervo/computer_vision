#include "ObjTracker.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <string>

using namespace lab6;

const std::array<cv::Scalar, ObjTracker::def_obj_num + 1> ObjTracker::colours
{
    cv::Scalar{ 0, 0, 255 },
    cv::Scalar{ 0, 255, 0 },
    cv::Scalar{ 255, 0, 0 },
    cv::Scalar{ 0, 255, 255 },
    cv::Scalar{ 255, 255, 255 }
};
const cv::Size ObjTracker::lk_win_size{ 11, 11 };
const cv::Point ObjTracker::frame_time_coord{ 15, 15 };
const cv::Scalar ObjTracker::frame_time_colour{ 0, 255, 0 };

ObjTracker::ObjTracker(
    cv::VideoCapture& vid,
    std::vector<ImgObject> objects
) :
    vid_{ vid },
    objects_(std::move(objects)),
    outputWin_{ win_name }
{}

bool ObjTracker::run()
{
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;
    using std::chrono::duration_cast;

    /* Initialise the first two video frames. */
    Log::info("Video initialisation.");
    cv::Mat prevFrame{};
    cv::Mat nextFrame{};
    vid_ >> prevFrame;
    vid_ >> nextFrame;

    /* Track the objects until the end of the video. */
    Log::info("Beginning object tracking.");
    while (!nextFrame.empty())
    {
        Log::info_d("*** NEW FRAME ***");
        auto startTime = high_resolution_clock::now();

        cv::Mat outputFrame{ nextFrame.clone() };

        for (int i = 0; i < objects_.size(); ++i)
        {
            Log::info_d("Object %d.", i);
            int colourId{ std::min(i, def_obj_num) };

            /* Compute the optical flow of the object. */
            std::vector<cv::Point2f> nextPts;
            std::vector<uchar> status;
            std::vector<float> error;

            Log::info_d("Computing optical flow.");
            cv::calcOpticalFlowPyrLK(prevFrame, nextFrame, objects_[i].features, nextPts, status, error, lk_win_size);

            /* Compute the error threshold. */
            float avgErr{ 0 };
            int count{ 0 };
            for (int j = 0; j < status.size(); ++j)
            {
                if (status[j] == 1)
                {
                    avgErr += error[i];
                    ++count;
                }
            }
            if (count == 0)
            {
                Log::error("No features found.");
                return false;
            }

            avgErr = std::max(avgErr / count, min_base_err);
            float errorTh{ avgErr * err_th_coeff };
            Log::info_d("Average error: %f.", avgErr);
            Log::info_d("Error threshold: %f", errorTh);

            /* Move object features and draw them. */
            std::vector<cv::Point2f> newFeatures;
            newFeatures.reserve(objects_[i].features.size());

            auto featIt = objects_[i].features.begin();
            for (int j = 0; j < status.size(); ++j)
            {
                if (status[j] == 1 && error[j] <= errorTh)
                {
                    cv::circle(
                        outputFrame,
                        newFeatures.emplace_back(nextPts[j]),
                        feature_radius,
                        colours[colourId],
                        feature_thickness
                    );
                    ++featIt;
                }
                else
                {
                    Log::warn("Erasing feature with error %f.", error[j]);
                    featIt = objects_[i].features.erase(featIt);
                }
            }
            if (newFeatures.empty())
            {
                Log::error("No features remaining for object %d.", colourId);
                return false;
            }
            Log::info_d(
                "Current features / Previous features: %d/%d.",
                newFeatures.size(),
                objects_[i].features.size()
            );

            /* Compute the new positions of the vertices. */
            Log::info_d("Computing homography.");
            cv::Mat mask;
            cv::Mat H{ cv::findHomography(objects_[i].features, newFeatures, mask, cv::RANSAC) };
            if (H.empty())
            {
                Log::error("Failed to compute homography matrix.");
                return false;
            }
            objects_[i].features = newFeatures;

            Log::info_d("Applying homography.");
            std::vector<cv::Point2f> newVertices;
            cv::perspectiveTransform(objects_[i].vertices, newVertices, H);
            objects_[i].vertices = newVertices;

            /* Move and draw the object's vertices. */
            Log::info_d("Drawing vertices.");
            for (int j = 0; j < objects_[i].vertices.size(); ++j)
            {
                cv::drawMarker(
                    outputFrame,
                    objects_[i].vertices[j],
                    colours[colourId],
                    cv::MARKER_CROSS,
                    20,
                    feature_thickness
                );
            }

            /* Draw the box that surrounds the object. */
            Log::info_d("Drawing boxes.");
            std::vector<cv::Point> vertices;
            vertices.reserve(4);
            for (const auto vertex : objects_[i].vertices)
            {
                vertices.emplace_back(vertex);
            }
            cv::polylines(outputFrame, vertices, true, colours[colourId], feature_thickness);
        }

        auto currentFrameTime = duration_cast<milliseconds>(high_resolution_clock::now() - startTime);

        std::string frameTimeStr{ "Frame time: " + std::to_string(currentFrameTime.count()) + "ms" };
        cv::putText(
            outputFrame,
            frameTimeStr,
            frame_time_coord,
            cv::HersheyFonts::FONT_HERSHEY_PLAIN,
            font_scale,
            frame_time_colour
        );

        outputWin_.showImg(outputFrame);
        prevFrame = nextFrame.clone();
        vid_ >> nextFrame;

        if (currentFrameTime <= frame_time) cv::waitKey(frame_time.count() - currentFrameTime.count());
        else cv::waitKey(1);
    }

    return true;
}

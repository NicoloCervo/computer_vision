#include "ObjTracker.hpp"

#include <algorithm>
#include <chrono>
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
const cv::Size ObjTracker::lk_win_size{ 9, 9 };
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
        auto startTime = high_resolution_clock::now();

        cv::Mat outputFrame{ nextFrame.clone() };

        int objId{ 0 };
        for (auto& obj : objects_)
        {
            int colourId{ std::clamp(objId, 0, def_obj_num) };
            /* Compute the optical flow of the object. */
            std::vector<cv::Point2f> nextPts;
            std::vector<uchar> status;
            std::vector<float> error;

            cv::calcOpticalFlowPyrLK(prevFrame, nextFrame, obj.features, nextPts, status, error, lk_win_size);

            /* Move object features and draw them. */
            std::vector<cv::Point2f> newFeatures;
            newFeatures.reserve(obj.features.size());

            auto featIt = obj.features.begin();
            for (int i = 0; i < status.size(); ++i)
            {
                if (status[i] == 1)
                {
                    cv::circle(
                        outputFrame,
                        newFeatures.emplace_back(nextPts[i]),
                        feature_radius,
                        colours[colourId]
                    );
                    ++featIt;
                }
                else
                {
                    featIt = obj.features.erase(featIt);
                }
            }
            if (newFeatures.empty())
            {
                Log::error("No features remaining for object %d.", colourId);
                return false;
            }

            /* Compute the new positions of the vertices. */
            cv::Mat mask;
            cv::Mat H{ cv::findHomography(obj.features, newFeatures, mask, cv::RANSAC) };
            obj.features = newFeatures;

            std::vector<cv::Point2f> newVertices;
            cv::perspectiveTransform(obj.vertices, newVertices, H);
            obj.vertices = newVertices;

            /* Move and draw the object's vertices. */
            for (int i = 0; i < obj.vertices.size(); ++i)
            {
                cv::drawMarker(
                    outputFrame,
                    obj.vertices[i],
                    colours[colourId],
                    cv::MARKER_CROSS
                );
            }

            /* Draw the box that surrounds the object. */
            std::vector<cv::Point> vertices;
            vertices.reserve(4);
            for (const auto vertex : obj.vertices)
            {
                vertices.emplace_back(vertex);
            }
            cv::polylines(outputFrame, vertices, true, colours[colourId]);

            ++objId;
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

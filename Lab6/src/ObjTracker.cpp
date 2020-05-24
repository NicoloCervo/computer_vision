#include "ObjTracker.hpp"

#include <algorithm>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace lab6;

const std::array<cv::Scalar, ObjTracker::def_obj_num + 1> ObjTracker::colours
{
    cv::Scalar{ 0, 0, 255 },
    cv::Scalar{ 0, 255, 0 },
    cv::Scalar{ 255, 0, 0 },
    cv::Scalar{ 0, 255, 255 },
    cv::Scalar{ 255, 255, 255 }
};

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
        cv::Mat outputFrame{ nextFrame.clone() };

        int objId{ 0 };
        for (auto& obj : objects_)
        {
            int colourId{ std::clamp(objId, 0, def_obj_num) };
            /* Compute the optical flow of the object. */
            std::vector<cv::Point2f> nextPts;
            std::vector<uchar> status;
            std::vector<float> error;

            cv::calcOpticalFlowPyrLK(prevFrame, nextFrame, obj.features, nextPts, status, error);

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

        outputWin_.showImg(outputFrame);
        prevFrame = nextFrame.clone();
        vid_ >> nextFrame;

        cv::waitKey(frame_delay);
    }

    return true;
}

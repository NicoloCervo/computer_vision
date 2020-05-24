#include "ObjTracker.hpp"

#include <algorithm>
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
    objects_{ objects },
    outputWin_{ win_name }
{}

void ObjTracker::run()
{
    /* Initialise the first two video frames. */
    cv::Mat prevFrame{};
    cv::Mat nextFrame{};
    vid_ >> prevFrame;
    vid_ >> nextFrame;

    /* Track the objects until the end of the video. */
    while (!nextFrame.empty())
    {
        cv::Mat outputFrame{ nextFrame.clone() };

        int objId{ 0 };
        for (auto obj : objects_)
        {
            int colourId{ std::clamp(objId, 0, def_obj_num) };
            /* Compute the optical flow of the object. */
            std::vector<cv::Point2f> nextPts;
            std::vector<unsigned char> status;
            std::vector<double> error;
            cv::calcOpticalFlowPyrLK(prevFrame, nextFrame, obj.features, nextPts, status, error);

            /* Move object features and draw them. */
            obj.features.clear();
            for (int i = 0; i < status.size(); ++i)
            {
                if (status[i] == 1)
                {
                    cv::circle(
                        outputFrame,
                        obj.features.emplace_back(nextPts[i]),
                        feature_radius,
                        colours[colourId]
                    );
                }
            }

            /* Compute the optical flow of the object's vertices. */
            cv::calcOpticalFlowPyrLK(prevFrame, nextFrame, obj.vertices, nextPts, status, error);

            /* Move and draw the object's vertices. */
            for (int i = 0; i < obj.vertices.size(); ++i)
            {
                if (status[i] == 1)
                {
                    obj.vertices[i] = nextPts[i];
                    cv::drawMarker(
                        outputFrame,
                        nextPts[i],
                        colours[colourId],
                        cv::MARKER_CROSS
                    );
                }
            }

            /* Draw the box that surrounds the object. */
            cv::polylines(outputFrame, obj.vertices, true, colours[colourId]);

            ++objId;
        }

        outputWin_.showImg(outputFrame);
        prevFrame = nextFrame;
        vid_ >> nextFrame;

        cv::waitKey(frame_delay);
    }
}

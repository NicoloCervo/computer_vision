#ifndef OBJTRACKER_HPP
#define OBJTRACKER_HPP

#include "lab6.hpp"

#include <array>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string_view>
#include <vector>

namespace lab6
{
    class ObjTracker
    {
    public:
        /********** CONSTANTS **********/
        static constexpr std::string_view win_name{ "Tracking" };
        static constexpr int frame_delay{ 50 };
        static constexpr int def_obj_num{ 4 };
        static constexpr int feature_radius{ 1 };
        static const std::array<cv::Scalar, def_obj_num + 1> colours;

        /********** CONSTRUCTOR **********/
        ObjTracker(
            cv::VideoCapture& vid,
            std::vector<ImgObject> objects
        );

        /********** METHODS **********/
        void run();

    private:
        cv::VideoCapture& vid_;
        std::vector<ImgObject> objects_;
        Window outputWin_;
    };
}

#endif
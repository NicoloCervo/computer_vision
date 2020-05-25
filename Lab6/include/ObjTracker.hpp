#ifndef OBJTRACKER_HPP
#define OBJTRACKER_HPP

#include "lab6.hpp"

#include <array>
#include <chrono>
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
        static constexpr std::chrono::milliseconds frame_time{ 33 };
        static constexpr int def_obj_num{ 4 };
        static constexpr int feature_radius{ 2 };
        static constexpr int feature_thickness{ 2 };
        static constexpr float err_th_coeff{ 6 };

        static const std::array<cv::Scalar, def_obj_num + 1> colours;
        static const cv::Size lk_win_size;
        static const cv::Point frame_time_coord;
        static constexpr double font_scale{ 1.0 };
        static const cv::Scalar frame_time_colour;

        /********** CONSTRUCTOR **********/
        ObjTracker(
            cv::VideoCapture& vid,
            std::vector<ImgObject> objects
        );

        /********** METHODS **********/
        bool run();

    private:
        cv::VideoCapture& vid_;
        std::vector<ImgObject> objects_;
        Window outputWin_;
    };
}

#endif
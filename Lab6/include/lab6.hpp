#ifndef LAB6_HPP
#define LAB6_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <array>
#include <cstdio>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

namespace lab6
{
    /********** CONSTANTS **********/
#ifdef LAB6_DEBUG_MSGS
    constexpr bool debug_msgs{ true };
#else
    constexpr bool debug_msgs{ false };
#endif

    constexpr double pi{ 3.14159265358979 };

    /********** ENUMS **********/
    /* Keycodes. */
    enum class Key
    {
        esc = 27
    };

    /********** STRUCT **********/
    template <typename T>
    struct Size
    {
        constexpr Size() = default;
        constexpr Size(const T a, const T b) :
            w{ a }, h{ b }
        {}

        T w{};
        T h{};
    };
    /* Rectangular object in an image. */
    struct ImgObject
    {
        std::vector<cv::Point2f> vertices{ 4 };
        std::vector<cv::Point2f> features;
    };

    /********** CLASSES **********/
    /* Basic console logging functions. */
    class Log
    {
    public:
        /********** CONSTANTS **********/
        static constexpr const char* asterisks =
            "********************************************************************************";
        static constexpr const char* spaces =
            "                                                                                ";
        static constexpr const char* hyphens =
            "--------------------------------------------------------------------------------";

        /********** METHODS **********/
        /* Standard versions, always active. */
        template <typename... Args>
        static void info(std::string_view msg, Args... args)
        {
            std::scoped_lock lck(mtx_);
            std::stringstream s;
            s << std::this_thread::get_id();
            std::printf("\033[1;37m[INFO - T%s] - ", s.str().c_str());
            std::printf(msg.data(), args...);
            std::printf("\n\033[0m");
        }
        template <typename... Args>
        static void warn(std::string_view msg, Args... args)
        {
            std::scoped_lock lck(mtx_);
            std::stringstream s;
            s << std::this_thread::get_id();
            std::printf("\033[0;33m[WARN - T%s] - ", s.str().c_str());
            std::printf(msg.data(), args...);
            std::printf("\n\033[0m");
        }
        template <typename... Args>
        static void error(std::string_view msg, Args... args)
        {
            std::scoped_lock lck(mtx_);
            std::stringstream s;
            s << std::this_thread::get_id();
            std::printf("\033[0;31m[ERROR - T%s] - ", s.str().c_str());
            std::printf(msg.data(), args...);
            std::printf("\n\033[0m");
        }
        template <typename... Args>
        static void fatal(std::string_view msg, Args... args)
        {
            std::scoped_lock lck(mtx_);
            std::stringstream s;
            s << std::this_thread::get_id();
            std::printf("\033[0;35m[FATAL - T%s] - ", s.str().c_str());
            std::printf(msg.data(), args...);
            std::printf("\n\033[0m");
        }

        /* Debug versions, only active if debug_msgs is true. */
        template <typename... Args>
        static void info_d(std::string_view msg, Args... args)
        {
            if constexpr (debug_msgs)
            {
                std::scoped_lock lck(mtx_);
                std::stringstream s;
                s << std::this_thread::get_id();
                std::printf("\033[1;37m[DINFO - T%s] - ", s.str().c_str());
                std::printf(msg.data(), args...);
                std::printf("\n\033[0m");
            }
        }
        template <typename... Args>
        static void warn_d(std::string_view msg, Args... args)
        {
            if constexpr (debug_msgs)
            {
                std::scoped_lock lck(mtx_);
                std::stringstream s;
                s << std::this_thread::get_id();
                std::printf("\033[0;33m[DWARN - T%s] - ", s.str().c_str());
                std::printf(msg.data(), args...);
                std::printf("\n\033[0m");
            }
        }
        template <typename... Args>
        static void error_d(std::string_view msg, Args... args)
        {
            if constexpr (debug_msgs)
            {
                std::scoped_lock lck(mtx_);
                std::stringstream s;
                s << std::this_thread::get_id();
                std::printf("\033[0;31m[DERROR - T%s] - ", s.str().c_str());
                std::printf(msg.data(), args...);
                std::printf("\n\033[0m");
            }
        }
        template <typename... Args>
        static void fatal_d(std::string_view msg, Args... args)
        {
            if constexpr (debug_msgs)
            {
                std::scoped_lock lck(mtx_);
                std::stringstream s;
                s << std::this_thread::get_id();
                std::printf("\033[0;35m[DFATAL - T%s] - ", s.str().c_str());
                std::printf(msg.data(), args...);
                std::printf("\n\033[0m");
            }
        }
    private:
        static std::mutex mtx_;
    };

    /* Basic wrapper for OpenCV windows. */
    class Window
    {
    public:
        /********** CONSTANTS **********/
        static constexpr size_t max_trckbar_num{ 8 }; // Maximum number of trackbars.

        /********** CONSTRUCTORS **********/
        explicit Window(std::string_view name);
        Window(const Window&) = delete;
        Window(Window&& win) = delete;
        ~Window();

        /********** OPERATORS **********/
        Window& operator=(const Window&) = delete;
        Window& operator=(Window&& win) = delete;

        /********** METHODS **********/
        bool addTrackBar(std::string_view name, int maxVal); // Add a trackbar with starting position 0.
        bool addTrackBar(std::string_view name, int startVal, int maxVal); // Add a trackbar.
        /* Fetch trackbar values if they changed since the last call. */
        [[nodiscard]] std::vector<int> fetchTrckVals(); // Return the current trackbar values.
        [[nodiscard]] bool modified() const; // Check whether the trackbar values changed.
        void showImg(const cv::Mat& img) const; // Show an image on the window.

    private:
        /********** METHODS **********/
        static void trckCallbck_(int val, void* ptr); // Trackbar callback.

        /********** VARIABLES **********/
        cv::String name_; // Name of the window.
        std::vector<int> trckBarVals_; // Values of the trackbars.
        bool trckModified_{ true }; // True if trackbar values changed since the last fetch.
    };
}

#endif
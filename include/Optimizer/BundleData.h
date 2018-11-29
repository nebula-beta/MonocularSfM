#ifndef __BUNDLE_DATA_H__
#define __BUNDLE_DATA_H__

#include <unordered_map>
#include <unordered_set>

#include <opencv2/opencv.hpp>

#include "Common/Types.h"

namespace MonocularSfM
{






class BundleData
{
public:
    struct Measurement
    {
        image_t image_id;
        cv::Vec2d point2D;

        Measurement(const image_t& image_id, const cv::Vec2d& point2D)
                    : image_id(image_id), point2D(point2D) { }

    };


    struct Landmark
    {
        cv::Vec3d point3D;
        std::vector<Measurement> measurements;

        Landmark() { }
        Landmark(const cv::Vec3d& point3D, const std::vector<Measurement>& measurements)
                : point3D(point3D), measurements(measurements) { }


    };

    struct CameraPose
    {
        cv::Mat rvec;
        cv::Mat tvec;

        CameraPose(){ }
        CameraPose(const cv::Mat& rvec, const cv::Mat& tvec)
                  : rvec(rvec), tvec(tvec) { }
    };


    cv::Mat K;
    std::unordered_map<point3D_t, Landmark> landmarks;
    std::unordered_map<image_t, CameraPose> camera_poses;
    std::unordered_set<image_t> constant_camera_pose;

    // 检查BA前后的误差
    double Debug();


};



} // MonocularSfM


#endif //__BUNDLE_DATA_H__

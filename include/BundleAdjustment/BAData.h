#ifndef __BA_DATA_H__
#define __BA_DATA_H__

#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "Common/Types.h"

namespace MonocularSfM
{





struct Measurement
{
    image_t image_id;
    cv::Point2f measurement;

    Measurement(image_t image_id, float x, float y) : image_id(image_id), measurement(cv::Point2f(x, y)){ }
    Measurement(image_t image_id, cv::Point2f measurement) : image_id(image_id), measurement(measurement) { }

};


struct Landmark
{
    point3D_t point3D_idx;
    cv::Point3f point3D;
    std::vector<Measurement> measurements;
    Landmark() { }
    Landmark(point3D_t point3D_idx, cv::Point3f point3D, std::vector<Measurement>& measurements)
        : point3D_idx(point3D_idx), point3D(point3D), measurements(measurements) { }


};

struct CameraPose
{
    image_t image_id;
    cv::Mat R;
    cv::Mat t;

    CameraPose(){ }
    CameraPose(image_t image_id, cv::Mat R, cv::Mat t) : image_id(image_id), R(R), t(t) { }
};

struct  BAData
{

    cv::Mat K;
    std::vector<Landmark> landmarks;
    std::unordered_map<image_t, CameraPose> camera_poses;
};







} // namespace MonocularSfM
#endif // __BA_DATA_H__

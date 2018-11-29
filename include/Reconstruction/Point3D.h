#ifndef __POINT3D_H__
#define __POINT3D_H__

#include <opencv2/opencv.hpp>
#include "Reconstruction/Track.h"

namespace MonocularSfM
{

class Point3D
{
public:
    Point3D();
    Point3D(const cv::Vec3d& xyz, const cv::Vec3b& color, const double& error = -1.0);


    const cv::Vec3d& XYZ() const;
    cv::Vec3d XYZ();
    void SetXYZ(const cv::Vec3d& xyz);

    const cv::Vec3b Color() const;
    cv::Vec3b Color();
    void SetColor(const cv::Vec3b& color);

    double Error() const;
    bool HasError() const;
    void SetError(const double& error);

    const class Track& Track() const;
    class Track Track();
    void SetTrack(const class Track& track);



    ////////////////////////////////////////////////////////////////////////////////
    // 向track中添加元素, 实际上是调用track的成员函数
    ////////////////////////////////////////////////////////////////////////////////
    void AddElement(const TrackElement& element);
    void AddElement(const image_t image_id, const point2D_t point2D_idx);
    void AddElements(const std::vector<TrackElement>& elements);

    ////////////////////////////////////////////////////////////////////////////////
    // 从track中删除元素, 实际上是调用track的成员函数
    ////////////////////////////////////////////////////////////////////////////////
    void DeleteElement(const size_t idx);
    void DeleteElement(const TrackElement& element);
    void DeleteElement(const image_t image_id, const point2D_t point2D_idx);

private:
    cv::Vec3d xyz_;
    cv::Vec3b color_;
    double error_;
    // track
    class Track track_;
};


} //namespace MonocularSfM

#endif //__POINT3D_H__

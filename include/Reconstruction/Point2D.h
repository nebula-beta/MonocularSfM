#ifndef __POINT2D_H__
#define __POINT2D_H__
#include <opencv2/opencv.hpp>
#include "Common/Types.h"
namespace MonocularSfM
{

class Point2D
{
public:
    Point2D();
    Point2D(const cv::Vec2d& xy, const cv::Vec3b& color, const point3D_t& point3D_id = INVALID);


	const cv::Vec2d& XY() const;
    cv::Vec2d XY();
    void SetXY(const cv::Vec2d& xy);

	const cv::Vec3b& Color() const;
    cv::Vec3b Color();
    void SetColor(const cv::Vec3b& color);

    const point3D_t& Point3DId() const;
    point3D_t Point3DId();

    void SetPoint3D(const point3D_t& point3D_id);
    void ResetPoint3D();
    bool HasPoint3D() const;

private:
    cv::Vec2d xy_;
    cv::Vec3b color_;
    point3D_t point3D_id_;
};


} //namespace MonocularSfM

#endif //__POINT2D_H__

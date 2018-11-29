#ifndef __IMAGE_H__
#define __IMAGE_H__
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "Common/Types.h"
#include "Reconstruction/Point2D.h"
namespace MonocularSfM
{

class Image
{
public:
    Image();
    Image(const image_t& image_id, const std::string& image_name);
    Image(const image_t& image_id,
          const std::string& image_name,
          const std::vector<Point2D>& points2D);

    const image_t& ImageId() const;
    image_t ImageId();
    void SetImageId(const image_t& image_id);

    const std::string& ImageName() const;
    std::string ImageName();
    void SetImageName(const std::string& image_name);



    point2D_t NumPoints2D() const;
    point2D_t NumPoints3D() const;

    const cv::Mat& Rotation() const;
    cv::Mat Rotation();
    void SetRotation(const cv::Mat& R);

    const cv::Mat& Translation() const;
    cv::Mat Translation();
    void SetTranslation(const cv::Mat& t);



    void SetPoints2D(const std::vector<Point2D>& points2D);
    const Point2D& GetPoint2D(const point2D_t& point2D_idx) const;
    Point2D GetPoint2D(const point2D_t& point2D_idx);

    ////////////////////////////////////////////////////////////////////////////////
    // 设置2D点所对应的3D点
    ////////////////////////////////////////////////////////////////////////////////
    void SetPoint2DForPoint3D(const point2D_t& point2D_idx,
                              const point3D_t& point3D_idx);

    void ResetPoint2DForPoint3D(const point2D_t& point2D_idx);

    bool Point2DHasPoint3D(const point2D_t& point2D_idx);



private:
    image_t image_id_;
    std::string image_name_;

    // 在track中的2D点的数量
	// SetPoint2DForPoint3D和ResetPoint2DForPoint3D会更新这个变量, +1或-1
    point2D_t num_points3D_;


    cv::Mat R_;
    cv::Mat t_;

    std::vector<Point2D> points2D_;

};


} //namespace MonocularSfM

#endif //__IMAGE_H__

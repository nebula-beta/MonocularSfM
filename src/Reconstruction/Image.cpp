#include "Reconstruction/Image.h"
using namespace MonocularSfM;

Image::Image()
{
    num_points3D_ = 0;
}
Image::Image(const image_t& image_id, const std::string& image_name)
            : image_id_(image_id), image_name_(image_name)
{
    num_points3D_ = 0;
}


Image::Image(const image_t& image_id,
      const std::string& image_name,
      const std::vector<Point2D>& points2D)
      : image_id_(image_id), image_name_(image_name),  points2D_(points2D)
{

}

const image_t& Image::ImageId() const
{
    return image_id_;
}
image_t Image::ImageId()
{
    return image_id_;
}
void Image::SetImageId(const image_t& image_id)
{
    image_id_ = image_id;
}

const std::string& Image::ImageName() const
{
    return image_name_;
}
std::string Image::ImageName()
{
    return image_name_;
}
void Image::SetImageName(const std::string& image_name)
{
    image_name_ = image_name;
}



point2D_t Image::NumPoints2D() const
{
    return points2D_.size();
}
point2D_t Image::NumPoints3D() const
{
    return num_points3D_;
}

const cv::Mat& Image::Rotation() const
{
    return R_;
}
cv::Mat Image::Rotation()
{
    return R_;
}
void Image::SetRotation(const cv::Mat& R)
{
    R_ = R;
}

const cv::Mat& Image::Translation() const
{
    return t_;
}
cv::Mat Image::Translation()
{
    return t_;
}
void Image::SetTranslation(const cv::Mat& t)
{
    t_ = t;
}

void Image::SetPoints2D(const std::vector<Point2D>& points2D)
{
    points2D_ = points2D;
}

const Point2D& Image::GetPoint2D(const point2D_t& point2D_idx) const
{
    assert(point2D_idx < points2D_.size());
    return points2D_[point2D_idx];
}
Point2D Image::GetPoint2D(const point2D_t& point2D_idx)
{
    assert(point2D_idx < points2D_.size());
    return points2D_[point2D_idx];
}

void Image::SetPoint2DForPoint3D(const point2D_t& point2D_idx,
                                 const point3D_t& point3D_idx)
{
    assert(point2D_idx < points2D_.size());
    num_points3D_ += 1;
    points2D_[point2D_idx].SetPoint3D(point3D_idx);
}

void Image::ResetPoint2DForPoint3D(const point2D_t& point2D_idx)
{
    assert(point2D_idx < points2D_.size());
    num_points3D_ -= 1;
    points2D_[point2D_idx].ResetPoint3D();
}

bool Image::Point2DHasPoint3D(const point2D_t& point2D_idx)
{
    assert(point2D_idx < points2D_.size());
    return points2D_[point2D_idx].HasPoint3D();
}

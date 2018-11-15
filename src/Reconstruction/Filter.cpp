#include "Reconstruction/Filter.h"

using namespace MonocularSfM;


////////////////////////////////////////////////////////////////////////////////
// RemoveWorldPtsByVisiable
////////////////////////////////////////////////////////////////////////////////
bool PointFilter::HasPositiveDepth(const cv::Point3f& point3d,
                                   const cv::Mat& R,
                                   const cv::Mat& t)
{
    assert(R.type() == CV_32F);
    assert(t.type() == CV_32F);

    cv::Mat pt = cv::Mat::zeros(3, 1, CV_32F);
    pt.at<float>(0, 0) = point3d.x;
    pt.at<float>(1, 0) = point3d.y;
    pt.at<float>(2, 0) = point3d.z;
    // 旋转平移到另一个相机坐标系
    pt = R * pt + t;
    if(pt.at<float>(2, 0) > std::numeric_limits<float>::epsilon())
    {
        return true;

    }

    return false;

}

bool PointFilter::RemoveWorldPtsByVisiable(const cv::Point3f& point3d,
                                           const cv::Mat& R1,
                                           const cv::Mat& t1,
                                           const cv::Mat& R2,
                                           const cv::Mat& t2)
{
    if(not HasPositiveDepth(point3d, R1, t1) ||
       not HasPositiveDepth(point3d, R2, t2))
    {

        return true;
    }
    return false;
}


size_t PointFilter::RemoveWorldPtsByVisiable(const std::vector<cv::Point3f>& point3ds,
                                             const cv::Mat& R1,
                                             const cv::Mat& t1,
                                             const cv::Mat& R2,
                                             const cv::Mat& t2,
                                             cv::Mat& inlier_mask)
{

    assert(R1.type() == CV_32F);
    assert(t1.type() == CV_32F);
    assert(R2.type() == CV_32F);
    assert(t2.type() == CV_32F);
    assert(inlier_mask.type() == CV_8U);


    size_t total_num = countNonZero(inlier_mask);
    size_t outlier_num = 0;

    for(int i = 0; i < inlier_mask.rows; ++i)
    {
        if(inlier_mask.at<uchar>(i ,0) == 0)
            continue;

        if(RemoveWorldPtsByVisiable(point3ds[i], R1, t1, R2, t2))
        {
            inlier_mask.at<uchar>(i, 0) = 0;
            outlier_num++;
        }
    }
    std::cout << "remove " << outlier_num << " outliers outof " << total_num << " with not visiable" << std::endl;
    return outlier_num;
}


////////////////////////////////////////////////////////////////////////////////
// RemoveWorldPtsByReprojectionError
////////////////////////////////////////////////////////////////////////////////
float PointFilter::CalculateReprojectionError(const cv::Point3f& point3D,
                                              const cv::Point2f& point2D,
                                              const cv::Mat& R,
                                              const cv::Mat& t,
                                              const cv::Mat& K)
{
    assert(R.type() == CV_32F);
    assert(t.type() == CV_32F);
    assert(K.type() == CV_32F);


    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    std::vector<cv::Point2f> proj_point2D;
    std::vector<cv::Point3f> point3Ds;
    point3Ds.push_back(point3D);

    cv::projectPoints(point3Ds, rvec, t, K, cv::noArray(), proj_point2D);

    // 2 norm
    double error = cv::norm(point2D - proj_point2D[0]);

    return error;
}


bool PointFilter::RemoveWorldPtsByReprojectionError(const cv::Point3f& point3d,
                                                    const cv::Point2f pt1,
                                                    const cv::Mat& R1,
                                                    const cv::Mat& t1,
                                                    const cv::Point2f pt2,
                                                    const cv::Mat& R2,
                                                    const cv::Mat& t2,
                                                    const cv::Mat& K,
                                                    double threshold_in_pixles)
{
    assert(R1.type() == CV_32F);
    assert(t1.type() == CV_32F);
    assert(R2.type() == CV_32F);
    assert(t2.type() == CV_32F);
    assert(K.type() == CV_32F);

    float error1 = CalculateReprojectionError(point3d, pt1, R1, t1, K);
    float error2 = CalculateReprojectionError(point3d, pt2, R2, t2, K);

    if(error1 > threshold_in_pixles ||
       error2 > threshold_in_pixles)
    {
        return true;
    }

    return false;
}


size_t PointFilter::RemoveWorldPtsByReprojectionError(const std::vector<cv::Point3f>& point3ds,
                                                      const std::vector<cv::Point2f>& pts1,
                                                      const cv::Mat& R1,
                                                      const cv::Mat& t1,
                                                      const std::vector<cv::Point2f>& pts2,
                                                      const cv::Mat& R2,
                                                      const cv::Mat& t2,
                                                      const cv::Mat& K,
                                                      cv::Mat& inlier_mask,
                                                      double threshold_in_pixles)
{


    assert(R1.type() == CV_32F);
    assert(t1.type() == CV_32F);
    assert(R2.type() == CV_32F);
    assert(t2.type() == CV_32F);
    assert(K.type() == CV_32F);
    assert(inlier_mask.type() == CV_8U);
    assert(point3ds.size() == inlier_mask.rows);
    assert(point3ds.size() == pts1.size());
    assert(pts1.size() == pts2.size());


    size_t total_num = countNonZero(inlier_mask);
    size_t outlier_num = 0;

    for(int i = 0; i < inlier_mask.rows; ++i)
    {
        if(inlier_mask.at<uchar>(i, 0) == 0)
            continue;
        if(RemoveWorldPtsByReprojectionError(point3ds[i], pts1[i], R1, t1, pts2[i], R2, t2, K,
                                             threshold_in_pixles))
        {
            inlier_mask.at<uchar>(i, 0) =  0;
            outlier_num++;
        }
    }

    std::cout << "remove " << outlier_num << " outliers outof " << total_num <<
                 " with repeojecion error bigger than " << (threshold_in_pixles) << std::endl;
    return outlier_num;
}



////////////////////////////////////////////////////////////////////////////////
// RemoveWorldPtsByParallaxAngle
////////////////////////////////////////////////////////////////////////////////


float PointFilter::CalculateParallaxAngle(const cv::Point3f& point3d,
                                          const cv::Point3f& proj_center1,
                                          const cv::Point3f& proj_center2)
{
    // (1)余弦定理
    // cosA = (b^2 + c^2 - a^2) / 2bc
    // (2)也可以使用a * b = |a| * |b| * cos来计算
    const float baseline = norm(proj_center1 - proj_center2);
    const float ray1 = norm(point3d - proj_center1);
    const float ray2 = norm(point3d - proj_center2);

    const float angle = std::abs(
            std::acos((ray1 * ray1 + ray2 * ray2 - baseline * baseline) / (2 * ray1 * ray2))
    );

    if(std::isnan(angle))
    {
        return 0;
    }
    else
    {
        return std::min<float>(angle, M_PI - angle) * 180 / M_PI;
    }
}

bool PointFilter::RemoveWorldPtsByParallaxAngle(const cv::Point3f& point3d,
                                                const cv::Mat& R1,
                                                const cv::Mat& t1,
                                                const cv::Mat& R2,
                                                const cv::Mat& t2,
                                                float threshold_in_angle_degree)
{
    assert(R1.type() == CV_32F);
    assert(t1.type() == CV_32F);
    assert(R2.type() == CV_32F);
    assert(t2.type() == CV_32F);

    // 光心在世界坐标系下的坐标
    cv::Mat O1 = -R1.t() * t1;
    cv::Mat O2 = -R2.t() * t2;

    cv::Point3f proj_center1(O1.at<float>(0, 0), O1.at<float>(1, 0), O1.at<float>(2, 0));
    cv::Point3f proj_center2(O2.at<float>(0, 0), O2.at<float>(1, 0), O2.at<float>(2, 0));

    float angle = CalculateParallaxAngle(point3d, proj_center1, proj_center2);
    if(angle < threshold_in_angle_degree)
    {
        return true;
    }
    return false;
}

size_t PointFilter::RemoveWorldPtsByParallaxAngle(const std::vector<cv::Point3f>& point3ds,
                                                  const cv::Mat& R1,
                                                  const cv::Mat& t1,
                                                  const cv::Mat& R2,
                                                  const cv::Mat& t2,
                                                  cv::Mat& inlier_mask,
                                                  float threshold_in_angle_degree)
{
    assert(R1.type() == CV_32F);
    assert(t1.type() == CV_32F);
    assert(R2.type() == CV_32F);
    assert(t2.type() == CV_32F);
    assert(inlier_mask.type() == CV_8U);
    assert(point3ds.size() == inlier_mask.rows);


    int total_num = countNonZero(inlier_mask);
    int outlier_num = 0;

    for(int i = 0; i < inlier_mask.rows; ++i)
    {
        if(inlier_mask.at<uchar>(i, 0) == 0)
            continue;



        if(RemoveWorldPtsByParallaxAngle(point3ds[i], R1, t1, R2, t2, threshold_in_angle_degree))
        {
            inlier_mask.at<uchar>(i, 0) = 0;
            outlier_num += 1;
        }
    }


    std::cout << "remove " << outlier_num << " outliers outof " << total_num <<
                 " with paralax angle less than " << (threshold_in_angle_degree) << std::endl;

    return outlier_num;
}








////////////////////////////////////////////////////////////////////////////////
// class TrackFilter
////////////////////////////////////////////////////////////////////////////////

int TrackFilter::filter_model_ = TrackFilter::STRICT_MODEL;
bool TrackFilter::RemoveTrack(const cv::Point3f& point3d,
                              const std::vector<cv::Point2f>& pts,
                              const std::vector<cv::Mat>& Rs,
                              const std::vector<cv::Mat>& ts,
                              const cv::Mat& K,
                              cv::Mat& inlier_mask,
                              double threshold_in_pixles,
                              double threshold_in_angle_degree,
                              int filter_model)
{

    assert(pts.size() == Rs.size());
    assert(Rs.size() == ts.size());
    assert(inlier_mask.rows == ts.size());
    assert(inlier_mask.type() == CV_8U);

    assert(countNonZero(inlier_mask) == inlier_mask.rows);
    assert(filter_model == LOOSE_MODEL || filter_model == SEMI_STRICT_MODEL || filter_model == STRICT_MODEL);

    filter_model_ = filter_model;

    bool is_filter = RemoveTrackPtsByVisiable(point3d, Rs, ts, inlier_mask);
    if(is_filter)
        return true;

    is_filter = RemoveTrackPtsByReprojectionError(point3d, pts, Rs, ts, K, inlier_mask, threshold_in_pixles);

    if(is_filter)
        return true;

//    is_filter = RemoveTrackPtsByAngle(point3d, pts, Rs, ts, K, inlier_mask, threshold_in_angle_degree);
    return false;
}


////////////////////////////////////////////////////////////////////////////////
// RemoveWorldPtsByVisiable
////////////////////////////////////////////////////////////////////////////////
bool TrackFilter::RemoveTrackPtsByVisiable(const cv::Point3f& point3d,
                                           const std::vector<cv::Mat>& Rs,
                                           const std::vector<cv::Mat>& ts,
                                           cv::Mat& inlier_mask)
{
    size_t outlier_num = 0;
    size_t total_num = countNonZero(inlier_mask);

    for(size_t i = 0; i < Rs.size(); ++i)
    {
        if(inlier_mask.at<uchar>(i, 0) == 0)
            continue;

        bool has_positive_depth = PointFilter::HasPositiveDepth(point3d, Rs[i], ts[i]);


        if(not has_positive_depth)
        {
            inlier_mask.at<uchar>(i, 0) = 0;
            outlier_num ++;
        }
    }

    return DecideToFilter(outlier_num, total_num, static_cast<size_t>(inlier_mask.rows));
}

////////////////////////////////////////////////////////////////////////////////
// RemoveWorldPtsByReprojectionError
////////////////////////////////////////////////////////////////////////////////
bool TrackFilter::RemoveTrackPtsByReprojectionError(const cv::Point3f& point3d,
                                                    const std::vector<cv::Point2f>& pts,
                                                    const std::vector<cv::Mat>& Rs,
                                                    const std::vector<cv::Mat>& ts,
                                                    const cv::Mat& K,
                                                    cv::Mat& inlier_mask,
                                                    double threshold_in_pixles)
{
    size_t outlier_num = 0;
    size_t total_num = countNonZero(inlier_mask);


    for(size_t i = 0; i < pts.size(); ++i)
    {
        if(inlier_mask.at<uchar>(i, 0) == 0)
            continue;
        float error = PointFilter::CalculateReprojectionError(point3d, pts[i], Rs[i], ts[i], K);

        if(error > threshold_in_pixles)
        {
            inlier_mask.at<uchar>(i, 0) = 0;
            outlier_num++;
        }
    }

    return DecideToFilter(outlier_num, total_num, static_cast<size_t>(inlier_mask.rows));
}



bool TrackFilter::RemoveTrackPtsByAngle(const cv::Point3d& point3d,
                                        const std::vector<cv::Point2f>& pts,
                                        const std::vector<cv::Mat>& Rs,
                                        const std::vector<cv::Mat>& ts,
                                        const cv::Mat& K,
                                        cv::Mat& inlier_mask,
                                        double threshold_in_degree)
{
    size_t outlier_num = 0;
    size_t total_num = countNonZero(inlier_mask);

    for(size_t i = 0; i < Rs.size(); ++i)
    {
        if(inlier_mask.at<uchar>(i, 0) == 0)
            continue;
        for(size_t j = i + 1; j < Rs.size(); ++j)
        {
            if(inlier_mask.at<uchar>(j, 0) == 0)
                continue;

            if(PointFilter::RemoveWorldPtsByParallaxAngle(point3d, Rs[i], ts[i], Rs[j], ts[j], threshold_in_degree))
            {
                inlier_mask.at<uchar>(i, 0) = 0;
                inlier_mask.at<uchar>(j, 0) = 0;
                outlier_num += 2;
            }
        }
    }
    return DecideToFilter(outlier_num, total_num, static_cast<size_t>(inlier_mask.rows));
}


bool TrackFilter::DecideToFilter(size_t outlier_num, size_t total_num, size_t rows)
{
    if(filter_model_ == STRICT_MODEL)
    {
        // 有一个点要过滤， 那么就过滤整个track
        return outlier_num > 0;
    }
    else if(filter_model_ == SEMI_STRICT_MODEL)
    {
        // 如果过滤的占超过0.5, 过滤
        if(static_cast<double>(outlier_num) / rows >= 0.5)
        {
            return true;
        }
        // 如果剩下的点不足2个， 过滤
        if(total_num - outlier_num < 2)
        {
            return true;
        }
        return false;
    }
    else
    {
        // 如果剩下的点不足2个， 过滤
        return (total_num - outlier_num) < 2;
    }
}


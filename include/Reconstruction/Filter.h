#ifndef __FILTER_H__
#define __FILTER_H__

#include <vector>
#include <opencv2/opencv.hpp>



namespace MonocularSfM
{



class PointFilter
{
public:
    constexpr static float max_3d_reprojection_error_in_pixels = 4.0f;
    constexpr static float min_parallax_in_degree = 2.0f;


    ////////////////////////////////////////////////////////////////////////////////
    // RemoveWorldPtsByVisiable
    ////////////////////////////////////////////////////////////////////////////////
    /**
     * 计算世界坐标系3D点point3d在另一个坐标系R,t下是否具有正深度
     * @param point3d   : 世界坐标系3D点
     * @param R         : 世界坐标系与另一个相机坐标系之间的旋转矩阵
     * @param t         : 世界坐标系与另一个相机坐标系之间的平移向量
     * @return          : 如果具有正深度， 返回true; 否则， 返回false.
     */
    static
    bool HasPositiveDepth(const cv::Point3f& point3d,
                          const cv::Mat& R,
                          const cv::Mat& t);

    /**
     * 根据世界坐标系3D点point3d是否在两个相机坐标系的前方（可见性约束）
     * 从而对3D进行过滤
     * @param point3d   : 世界坐标系3D点
     * @param R1        : 世界坐标系与第一个相机坐标系之间的旋转矩阵
     * @param t1        : 世界坐标系与第一个相机坐标系之间的平移向量
     * @param R2        : 世界坐标系与第二个相机坐标系之间的旋转矩阵
     * @param t2        : 世界坐标系与第二个相机坐标系之间的平移向量
     * @return          : 如果要对该点进行过滤， 返回true; 否则， 返回false.
     */
    static
    bool RemoveWorldPtsByVisiable(const cv::Point3f& point3d,
                                  const cv::Mat& R1,
                                  const cv::Mat& t1,
                                  const cv::Mat& R2,
                                  const cv::Mat& t2);
    /**
     * 根据可见性约束对3D点points进行过滤
     * @param point3ds      : 世界坐标系3D点集合
     * @param R1            : 世界坐标系与第一个相机坐标系之间的旋转矩阵
     * @param t1            : 世界坐标系与第一个相机坐标系之间的平移向量
     * @param R2            : 世界坐标系与第二个相机坐标系之间的旋转矩阵
     * @param t2            : 世界坐标系与第二个相机坐标系之间的平移向量
     * @param inlier_mask   : [input/output] 用来标记要对哪些点进行过滤， 以及过滤之后哪些点是内点.
     * @return              : 如果要对该点进行过滤， 返回true; 否则， 返回false.
     */
    static
    size_t RemoveWorldPtsByVisiable(const std::vector<cv::Point3f>& point3ds,
                                    const cv::Mat& R1,
                                    const cv::Mat& t1,
                                    const cv::Mat& R2,
                                    const cv::Mat& t2,
                                    cv::Mat& inlier_mask);


    ////////////////////////////////////////////////////////////////////////////////
    // RemoveWorldPtsByReprojectionError
    ////////////////////////////////////////////////////////////////////////////////
    /**
     * 计算世界坐标系3D点point3d投影到图像上后，与2D点pt的误差（重投影误差）
     * @param point3d   : 世界坐标系3D点
     * @param pt        : 观察到该3D点的2D图像点
     * @param R         : 世界坐标系与另一个相机坐标系之间的旋转矩阵
     * @param t         : 世界坐标系与另一个相机坐标系之间的平移向量
     * @param K         : 相机的内参矩阵
     * @return          : 返回3D点的投影点与2D图像点之间的2-norm(即距离的平方开根号）.
     */
    static
    float CalculateReprojectionError(const cv::Point3f& point3D,
                                     const cv::Point2f& point2D,
                                     const cv::Mat& R,
                                     const cv::Mat& t,
                                     const cv::Mat& K);
    /**
     * 根据世界坐标系3D点point3d在两个相机坐标系下的重投影误差
     * 来决定是否对该3D点进行过滤
     * @param point3d               : 世界坐标系3D点
     * @param pt1                   : 第一张图片观察到该3D点的2D图像点
     * @param R1                    : 世界坐标系与第一个相机坐标系之间的旋转矩阵
     * @param t1                    : 世界坐标系与第一个相机坐标系之间的平移向量
     * @param pt2                   : 第二张图片观察到该3D点的2D图像点
     * @param R2                    : 世界坐标系与第二个相机坐标系之间的旋转矩阵
     * @param t2                    : 世界坐标系与第二个相机坐标系之间的平移向量
     * @param K                     : 相机的内参矩阵
     * @param threshold_in_pixles   : 重投影误差阈值
     * @return                      : 如果要对该点进行过滤， 返回true；否则，返回false.
     */
    static
    bool RemoveWorldPtsByReprojectionError(const cv::Point3f& point3d,
                                           const cv::Point2f pt1,
                                           const cv::Mat& R1,
                                           const cv::Mat& t1,
                                           const cv::Point2f pt2,
                                           const cv::Mat& R2,
                                           const cv::Mat& t2,
                                           const cv::Mat& K,
                                           double threshold_in_pixles = max_3d_reprojection_error_in_pixels);


    /**
     * 根据世界坐标系3D点point3d在两个相机坐标系下的重投影误差
     * 来决定是否对该3D点进行过滤
     * @param point3ds              : 世界坐标系3D点集合
     * @param pts1                  : 第一张图片观察到3D点集合的2D图像点集合
     * @param R1                    : 世界坐标系与第一个相机坐标系之间的旋转矩阵
     * @param t1                    : 世界坐标系与第一个相机坐标系之间的平移向量
     * @param pts2                  : 第二张图片观察到3D点集合的2D图像点集合
     * @param R2                    : 世界坐标系与第二个相机坐标系之间的旋转矩阵
     * @param t2                    : 世界坐标系与第二个相机坐标系之间的平移向量
     * @param K                     : 相机的内参矩阵
     * @param inlier_mask           : [input/output] 用来标记要对哪些点进行过滤， 以及过滤之后哪些点是内点.
     * @param threshold_in_pixles   : 重投影误差阈值
     * @return                      : 如果要对该点进行过滤， 返回true；否则，返回false.
     */


    static
    size_t RemoveWorldPtsByReprojectionError(const std::vector<cv::Point3f>& point3ds,
                                             const std::vector<cv::Point2f>& pts1,
                                             const cv::Mat& R1,
                                             const cv::Mat& t1,
                                             const std::vector<cv::Point2f>& pts2,
                                             const cv::Mat& R2,
                                             const cv::Mat& t2,
                                             const cv::Mat& K,
                                             cv::Mat& inlier_mask,
                                             double threshold_in_pixles = max_3d_reprojection_error_in_pixels);

    ////////////////////////////////////////////////////////////////////////////////
    // RemoveWorldPtsByParallaxAngle
    ////////////////////////////////////////////////////////////////////////////////
    /**
     * 计算向量point3d - proj_center1与向量point3d - proj_center2之间的夹角
     * @param point3d           : 世界坐标系3D点
     * @param proj_center1      : 第一个相机的光心在世界坐标系下的坐标
     * @param proj_center2      : 第二个相机的光心在世界坐标系下的坐标
     * @return                  : 返回向量point3d - proj_center1与向量point3d - proj_center2之间的夹角.
     */
    static
    float CalculateParallaxAngle(const cv::Point3f& point3d,
                                 const cv::Point3f& proj_center1,
                                 const cv::Point3f& proj_center2);
    /**
     * 根据视差角来决定是否对该3D点进行过滤
     * @param point3d                    : 世界坐标系3D点
     * @param R1                         : 世界坐标系与第一个相机坐标系之间的旋转矩阵
     * @param t1                         : 世界坐标系与第一个相机坐标系之间的平移向量
     * @param R2                         : 世界坐标系与第二个相机坐标系之间的旋转矩阵
     * @param t2                         : 世界坐标系与第二个相机坐标系之间的平移向量
     * @return                           : 如果要对该点进行过滤， 返回true; 否则， 返回false.
     * @param threshold_in_angle_degree  : 视差角阈值， 默认设为2度.
     * @return                           : 如果要对该点进行过滤， 返回true；否则，返回false.
     */
    static
    bool RemoveWorldPtsByParallaxAngle(const cv::Point3f& point3d,
                                       const cv::Mat& R1,
                                       const cv::Mat& t1,
                                       const cv::Mat& R2,
                                       const cv::Mat& t2,
                                       float threshold_in_angle_degree = min_parallax_in_degree);
    /**
     * 根据视差角来决定是否对该3D点进行过滤
     * @param point3ds                   : 世界坐标系3D点集合
     * @param R1                         : 世界坐标系与第一个相机坐标系之间的旋转矩阵
     * @param t1                         : 世界坐标系与第一个相机坐标系之间的平移向量
     * @param R2                         : 世界坐标系与第二个相机坐标系之间的旋转矩阵
     * @param t2                         : 世界坐标系与第二个相机坐标系之间的平移向量
     * @param inlier_mask                : [input/output] 用来标记要对哪些点进行过滤， 以及过滤之后哪些点是内点.
     * @param threshold_in_angle_degree  : 视差角阈值， 默认设为2度.
     * @return                           : 如果要对该点进行过滤， 返回true；否则，返回false.
     */
    static
    size_t RemoveWorldPtsByParallaxAngle(const std::vector<cv::Point3f>& point3ds,
                                         const cv::Mat& R1,
                                         const cv::Mat& t1,
                                         const cv::Mat& R2,
                                         const cv::Mat& t2,
                                         cv::Mat& inlier_mask,
                                         float threshold_in_angle_degree = min_parallax_in_degree);

};





class TrackFilter
{
public:
    constexpr static float max_3d_reprojection_error_in_pixels = PointFilter::max_3d_reprojection_error_in_pixels;
    constexpr static float min_parallax_in_degree = PointFilter::min_parallax_in_degree;

    /**
     *  LOOSE_MODEL             : 当track中的点被过滤得不足两个时， 删除该track
     *  SEME_STRICT_MODEL       : 当track中的点被过滤超过一半时， 删除该track
     *  STRICT_MODEL            : 只要track中有一个点被过滤， 删除该track
     */
    enum{
        LOOSE_MODEL       = 0,
        SEMI_STRICT_MODEL = 1,
        STRICT_MODEL      = 2
    };

public:

    /**
     * 根据过滤模型，和RemoveTrackPtsByVisiable和RemoveTrackPtsByReprojectionError过滤结果
     * 来决定对track进行删除还是过滤
     * @param point3d                   : 世界坐标系3D点
     * @param pts                       : 观察到该3D点的2D点集合
     * @param Rs                        : 2D点观察点所在图像的旋转矩阵
     * @param ts                        : 2D点观察点所在图像的平移向量
     * @param K                         : 相机内参矩阵
     * @param inlier_mask               : [input/output] 用来标记要对哪些点进行过滤， 以及过滤之后哪些点是内点.
     * @param threshold_in_pixles       : 重投影误差阈值
     * @param threshold_in_angle_degree : 视差角阈值 //TODO
     * @return                          : 如果要删除rack， 返回true；否则，返回false.
     *                                  : 如果不要删除rack, 那么根据inlier_mask的结果对track进行过滤
     */
    static
    bool RemoveTrack(const cv::Point3f& point3d,
                     const std::vector<cv::Point2f>& pts,
                     const std::vector<cv::Mat>& Rs,
                     const std::vector<cv::Mat>& ts,
                     const cv::Mat& K,
                     cv::Mat& inlier_mask,
                     double threshold_in_pixles = max_3d_reprojection_error_in_pixels,
                     double threshold_in_angle_degree = min_parallax_in_degree,
                     int filter_model = LOOSE_MODEL);

private:
    ////////////////////////////////////////////////////////////////////////////////
    // RemoveWorldPtsByVisiable
    ////////////////////////////////////////////////////////////////////////////////
    /**
     * 根据可见性对track进行过滤
     * @param point3d        : 世界坐标系3D点
     * @param Rs             : 2D点观察点所在图像的旋转矩阵
     * @param ts             : 2D点观察点所在图像的平移向量
     * @return               : 如果要删除rack， 返回true；否则，返回false.
     *                       : 如果不要删除rack, 那么根据inlier_mask的结果对track进行过滤
     */
    static
    bool RemoveTrackPtsByVisiable(const cv::Point3f& point3d,
                                  const std::vector<cv::Mat>& Rs,
                                  const std::vector<cv::Mat>& ts,
                                  cv::Mat& inlier_mask);
    ////////////////////////////////////////////////////////////////////////////////
    // RemoveWorldPtsByReprojectionError
    ////////////////////////////////////////////////////////////////////////////////
    /**
     * 根据重投影误差对track进行过滤
     * @param point3d                 : 世界坐标系3D点
     * @param pts                     : 观察到该3D点的2D点集合
     * @param Rs                      : 2D点观察点所在图像的旋转矩阵
     * @param ts                      : 2D点观察点所在图像的平移向量
     * @param K                       : 相机内参矩阵
     * @param inlier_mask             : [input/output] 用来标记要对哪些点进行过滤， 以及过滤之后哪些点是内点.
     * @param threshold_in_pixles     : 重投影误差阈值
     * @return                        : 如果要删除rack， 返回true；否则，返回false.
     *                                : 如果不要删除rack, 那么根据inlier_mask的结果对track进行过滤
     */
    static
    bool RemoveTrackPtsByReprojectionError(const cv::Point3f& point3d,
                                           const std::vector<cv::Point2f>& pts,
                                           const std::vector<cv::Mat>& Rs,
                                           const std::vector<cv::Mat>& ts,
                                           const cv::Mat& K,
                                           cv::Mat& inlier_mask,
                                           double threshold_in_pixles);



    /**
     * @brief RemoveTrackPtsByAngle
     * @param point3d
     * @param pts
     * @param Rs
     * @param ts
     * @param K
     * @param inlier_mask
     * @param threshold_in_degree
     * @return
     */
    static
    bool RemoveTrackPtsByAngle(const cv::Point3d& point3d,
                               const std::vector<cv::Point2f>& pts,
                               const std::vector<cv::Mat>& Rs,
                               const std::vector<cv::Mat>& ts,
                               const cv::Mat& K,
                               cv::Mat& inlier_mask,
                               double threshold_in_degree);

    /**
     * 根据过滤模式，决定是否对当前的track进行过滤
     * @param outlier_num   :
     * @param total_num     :
     * @param rows          :
     * @return              :  如果要删除rack， 返回true；否则，返回false.
     */
    static
    bool DecideToFilter(size_t outlier_num, size_t total_num, size_t rows);

//    bool RemoveTrackPtsByParallaxAngle(const Point3f& point3d,
//                                       const std::vector<cv::Mat>& Rs,
//                                       const std::vector<cv::Mat>& ts,
//                                       cv::Mat& inlier_mask,
//                                       double threshold_in_angle_degree);
public:
    static int filter_model_;
};






} // namespace Undefine



#endif // __FILTER_H__

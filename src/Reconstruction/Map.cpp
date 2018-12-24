#include "Reconstruction/Map.h"
#include "Reconstruction/Projection.h"

#include "Exportor/OpenMVSInterface.h"

#include <sys/stat.h>



using namespace MonocularSfM;


// 数据库中的特征点是没有去畸变的， 所以要进行去畸变处理
void UndistortFeature(const cv::Mat& K,
                      const cv::Mat& dist_coef,
                      const std::vector<cv::Point2f>& points2D,
                      std::vector<cv::Point2f>& undistort_points2D)
{
    cv::Mat mat(points2D.size(), 2, CV_32F);
    for(size_t i = 0; i < points2D.size(); ++i)
    {
        mat.at<float>(i, 0) = points2D[i].x;
        mat.at<float>(i, 1) = points2D[i].y;
    }

    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, K, dist_coef, cv::Mat(), K);
    mat = mat.reshape(1);

    undistort_points2D.resize(points2D.size());

    for(size_t i = 0; i < points2D.size(); ++i)
    {
        cv::Point2f undistort_point2D(mat.at<float>(i, 0), mat.at<float>(i, 1));
        undistort_points2D[i] = undistort_point2D;
    }
}




Map::Map(cv::Ptr<SceneGraph> scene_graph, const cv::Mat& K, const cv::Mat& dist_coef)
    : scene_graph_(scene_graph), K_(K),  dist_coef_(dist_coef)
{
    num_point3D_idx_ = 0;
    assert(K.type() == CV_64F);
    assert(dist_coef.type() == CV_64F);
}

void Map::Load(cv::Ptr<Database> database)
{
    std::vector<Database::Image> db_images = database->ReadAllImages();
    for(const auto& db_image : db_images)
    {

        images_[db_image.id].SetImageId(db_image.id);
        images_[db_image.id].SetImageName(db_image.name);

        const std::vector<cv::KeyPoint>& kpts =  database->ReadKeyPoints(db_image.id);
        const std::vector<cv::Vec3b>& colors = database->ReadKeyPointsColor(db_image.id);

        std::vector<cv::Point2f> pts;
        cv::KeyPoint::convert(kpts, pts);

        std::vector<cv::Point2f> undistort_pts;
        if(dist_coef_.at<double>(0, 0) != 0.0)
        {
            UndistortFeature(K_, dist_coef_, pts, undistort_pts);
            pts = undistort_pts;

        }

        std::vector<Point2D> points2D(pts.size());
        for(size_t i = 0; i < pts.size(); ++i)
        {
            Point2D point2D(cv::Vec2d(pts[i].x, pts[i].y), colors[i]);
            points2D[i] = point2D;
        }

        images_[db_image.id].SetPoints2D(points2D);
    }
}

void Map::AddImagePose(const image_t& image_id,
                       const cv::Mat& R,
                       const cv::Mat& t)
{

    images_[image_id].SetRotation(R);
    images_[image_id].SetTranslation(t);
    registered_images_.push_back(image_id);
    registered_.insert(image_id);
    ClearModifiedPoint3DIds();
}


point3D_t Map::AddPoint3D(const cv::Vec3d& xyz, const Track& track)
{

    double error = ComputeTrackError(xyz, track);
    cv::Vec3b color = ComputeTrackColor(track);

    return AddPoint3D(xyz, track, color, error);
}

point3D_t Map::AddPoint3D(const cv::Vec3d& xyz, const Track& track, const double&error)
{
    int sum_color_x = 0;
    int sum_color_y = 0;
    int sum_color_z = 0;

    for(const TrackElement& element : track.Elements())
    {
        const Image& image = images_[element.image_id];
        const cv::Vec3b& color = image.GetPoint2D(element.point2D_idx).Color();
        sum_color_x += color(0);
        sum_color_y += color(1);
        sum_color_z += color(2);
    }

    cv::Vec3b color(static_cast<uchar>(sum_color_x / track.Length()),
                    static_cast<uchar>(sum_color_y / track.Length()),
                    static_cast<uchar>(sum_color_z / track.Length()));

    return AddPoint3D(xyz, track, color, error);

}
point3D_t Map::AddPoint3D(const cv::Vec3d& xyz, const Track& track,
                          const cv::Vec3b& color, const double&error)
{
    Point3D point3D;
    point3D.SetXYZ(xyz);
    point3D.SetTrack(track);

    point3D.SetColor(color);
    point3D.SetError(error);
    points3D_[num_point3D_idx_] = point3D;
    modified_point3D_ids_.insert(num_point3D_idx_);


    // 设置图像的2D点能够看到该3D点
    for(const TrackElement& element : track.Elements())
    {
        images_[element.image_id].SetPoint2DForPoint3D(element.point2D_idx, num_point3D_idx_);
    }

    return num_point3D_idx_ ++;
}





void Map::RemovePoint3D(const point3D_t& point3D_idx)
{
    assert(HasPoint3DInMap(point3D_idx));

    const Point3D& point3D = points3D_[point3D_idx];
    const Track& track = point3D.Track();
//    points3D_.erase(point3D_idx); // 不能放在这里，因为删除之后， 上面的那些引用就变得无效了

    // 设置图像的2D点， 使其不能看到3D点
    for(const TrackElement& element : track.Elements())
    {
        images_[element.image_id].ResetPoint2DForPoint3D(element.point2D_idx);
    }

    points3D_.erase(point3D_idx);

}


void Map::AddObservation(const point3D_t& point3D_idx, const TrackElement& track_el, const double& error)
{
    assert(HasPoint3DInMap(point3D_idx));

    const Image& image = images_[track_el.image_id];
    Point3D& point3D = points3D_[point3D_idx];


    double ave_error = (point3D.Error() * point3D.Track().Length() + error) / (point3D.Track().Length() + 1);
    const cv::Vec3b& point3D_old_color = point3D.Color();
    const cv::Vec3b& color = image.GetPoint2D(track_el.point2D_idx).Color();

    int point3D_new_color_x = (point3D_old_color(0) * point3D.Track().Length() + color(0)) / (point3D.Track().Length() + 1);
    int point3D_new_color_y = (point3D_old_color(1) * point3D.Track().Length() + color(1)) / (point3D.Track().Length() + 1);
    int point3D_new_color_z = (point3D_old_color(2) * point3D.Track().Length() + color(2)) / (point3D.Track().Length() + 1);

    point3D.SetColor(cv::Vec3b(point3D_new_color_x, point3D_new_color_y, point3D_new_color_z));
    point3D.SetError(ave_error);
    point3D.AddElement(track_el);

#ifdef DEBUG
    const Track& track = point3D.Track();
    double real_error = ComputeTrackError(point3D.XYZ(), track);
//    std::cout << "real error : " << real_error << ", error : " << point3D.Error() << ", ave error : " << ave_error << std::endl;
//    std::cout << "--------------------------------------------" << std::endl;
    assert(std::fabs(real_error - point3D.Error()) < 1e-6);
#endif

    // 设置图像的2D点能够看到该3D点
    images_[track_el.image_id].SetPoint2DForPoint3D(track_el.point2D_idx, point3D_idx);
}
void Map::RemoveObservation(const point3D_t& point3D_idx, const TrackElement& track_el)
{
    assert(HasPoint3DInMap(point3D_idx));

    // 设置图像的2D点， 使其不能看到3D点
    images_[track_el.image_id].ResetPoint2DForPoint3D(track_el.point2D_idx);

    points3D_[point3D_idx].DeleteElement(track_el);
    // TODO,  下面这些， 已经在FilterPoints3DWithLargeReprojectionError函数中实现了
    // 不知道要不要移到这里来???
//    point3D.SetColor();
//    point3D.SetError()

}


std::vector<image_t> Map::GetNextImageIds()
{
    std::unordered_map<image_t, size_t> next_images_score;
    for(const auto& point3D_el : points3D_)
    {
        const point3D_t& point3D_idx = point3D_el.first;
        const Point3D& point3D = point3D_el.second;
        const Track& track = point3D.Track();
        for(const TrackElement& track_el : track.Elements())
        {
            const image_t& image_id = track_el.image_id;
            const point2D_t& point2D_idx = track_el.point2D_idx;
            std::vector<SceneGraph::Correspondence> corrs =
                    scene_graph_->FindCorrespondences(image_id, point2D_idx);

            for(const auto& corr : corrs)
            {


                if(registered_.count(corr.image_id) == 0)
                {
                    next_images_score[corr.image_id] += 1;
                }

            }
        }
    }

    struct ImageInfo
    {
        image_t image_id;
        size_t score;
    };

    std::vector<ImageInfo> image_infos;
    for(const auto& next_image_score : next_images_score)
    {

        ImageInfo image_info;
        image_info.image_id = next_image_score.first;
        image_info.score = next_image_score.second;
        image_infos.push_back(image_info);
    }

    std::sort(
          image_infos.begin(), image_infos.end(),
          [](const ImageInfo& image_info1, const ImageInfo& image_info2)
            {
                return image_info1.score > image_info2.score;
            }
           );

    std::vector<image_t> image_ids;

    for(const ImageInfo& image_info : image_infos)
    {
        image_ids.push_back(image_info.image_id);
    }
    return image_ids;

}


point2D_t Map::NumPoints2DInImage(const image_t& image_id)
{
    return images_[image_id].NumPoints2D();
}
bool Map::HasPoint3DInMap(const point3D_t& point3D_idx)
{
    return points3D_.count(point3D_idx) != 0;
}
bool Map::HasPoint3DInImage(const image_t& image_id, const point2D_t& point2D_idx)
{
    assert(images_.count(image_id) != 0);

    return images_[image_id].Point2DHasPoint3D(point2D_idx);
}
Point3D Map::GetPoint3DInImage(const image_t& image_id, const point2D_t& point2D_idx)
{
    assert(HasPoint3DInImage(image_id, point2D_idx));

    point3D_t point3D_idx = images_[image_id].GetPoint2D(point2D_idx).Point3DId();
    return points3D_[point3D_idx];
}


size_t Map::NumRegisteredImage()
{
    return registered_images_.size();
}



void Map::Get2D2DCorrespoindencesBetweenImages(const image_t& image_id1,
                                               const image_t& image_id2,
                                               std::vector<cv::Vec2d>& points2D1,
                                               std::vector<cv::Vec2d>& points2D2,
                                               std::vector<point2D_t>& point2D_idxs1,
                                               std::vector<point2D_t>& point2D_idxs2)
{
    const std::vector<cv::DMatch>& matches = scene_graph_->FindCorrespondencesBetweenImages(image_id1, image_id2);

    points2D1.resize(matches.size());
    points2D2.resize(matches.size());
    point2D_idxs1.resize(matches.size());
    point2D_idxs2.resize(matches.size());

    const Image& image1 = images_[image_id1];
    const Image& image2 = images_[image_id2];
    for(size_t i = 0; i < matches.size(); ++i)
    {
        const point2D_t& point2D_idx1 = matches[i].queryIdx;
        const point2D_t& point2D_idx2 = matches[i].trainIdx;

        cv::Vec2d cv_point2D1 = image1.GetPoint2D(point2D_idx1).XY();
        cv::Vec2d cv_point2D2 = image2.GetPoint2D(point2D_idx2).XY();

        points2D1[i] = cv_point2D1;
        points2D2[i] = cv_point2D2;
        point2D_idxs1[i] = point2D_idx1;
        point2D_idxs2[i] = point2D_idx2;
    }
}
void Map::Get2D3DCorrespondences(const image_t& image_id,
                                std::vector<cv::Vec2d>& points2D,
                                std::vector<cv::Vec3d>& points3D,
                                std::vector<point2D_t>& point2D_idxs,
                                std::vector<point3D_t>& point3D_idxs)
{
    assert(images_.count(image_id) != 0);

    points2D.clear();
    points3D.clear();
    point2D_idxs.clear();
    point3D_idxs.clear();
    const Image& image = images_[image_id];




    std::unordered_map<point2D_t, std::unordered_set<point3D_t>> visit;
    for(point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx)
    {
        Point2D point2D = image.GetPoint2D(point2D_idx);

        std::vector<SceneGraph::Correspondence> corrs =  scene_graph_->FindCorrespondences(image_id, point2D_idx);

        for(const auto& corr : corrs)
        {
            assert(images_.count(corr.image_id) != 0);

            const Image& other_image = images_[corr.image_id];



            if(other_image.GetPoint2D(corr.point2D_idx).HasPoint3D())
//            if(HasPoint3DInImage(corr.image_id, corr.point2D_idx))
            {
                point3D_t point3D_idx = other_image.GetPoint2D(corr.point2D_idx).Point3DId();

                // TODO : 改善这个判断， 目的 : 省空间
                if(visit.count(point2D_idx) != 0 && visit[point2D_idx].count(point3D_idx) != 0)
                {
                    // 已经有了该2D-3D对应
                    continue;
                }

                visit[point2D_idx].insert(point3D_idx);

                Point3D point3D = points3D_[point3D_idx];

                points2D.push_back(point2D.XY());
                points3D.push_back(point3D.XYZ());
                point2D_idxs.push_back(point2D_idx);
                point3D_idxs.push_back(point3D_idx);
            }
        }

    }
}

void Map::Get2D2DCorrespondences(const image_t& image_id,
                                 std::vector<std::vector<CorrData>>& points2D_corr_datas)
{
    assert(images_.count(image_id) != 0);

    points2D_corr_datas.clear();
    const Image& image = images_[image_id];

    for(point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx)
    {
        if(HasPoint3DInImage(image_id, point2D_idx))
            continue;

        // two view observation表示
        // 图像1的特征点i 只与 图像2的特征点j发生了匹配
        // 图像2的特征点j 也同样只与 图像1的特征点i发生了匹配
        // 这样的特征点对于重建是没有帮助的
        // 不同对其它图像提供2D-3D对应
        if(scene_graph_->IsTwoViewObservation(image_id, point2D_idx))
            continue;

        Point2D point2D = image.GetPoint2D(point2D_idx);

        std::vector<SceneGraph::Correspondence> corrs =  scene_graph_->FindCorrespondences(image_id, point2D_idx);

        std::vector<CorrData> corr_datas;
        for(const auto& corr : corrs)
        {
            assert(images_.count(corr.image_id) != 0);

            const Image& corr_image = images_[corr.image_id];
            const Point2D& corr_point2D = corr_image.GetPoint2D(corr.point2D_idx);

            if(corr_image.GetPoint2D(corr.point2D_idx).HasPoint3D())
            {
                continue;
            }
            CorrData corr_data;
            corr_data.image_id = corr.image_id;
            corr_data.point2D_idx = corr.point2D_idx;
            corr_data.R = corr_image.Rotation();
            corr_data.t = corr_image.Translation();
            corr_data.point2D = corr_point2D.XY();
            corr_datas.push_back(corr_data);
        }
        if(!corr_datas.empty())
        {
            CorrData corr_data;
            corr_data.image_id = image.ImageId();
            corr_data.point2D_idx = point2D_idx;
            corr_data.R = image.Rotation();
            corr_data.t = image.Translation();
            corr_data.point2D = point2D.XY();
            corr_datas.push_back(corr_data);

            points2D_corr_datas.push_back(corr_datas);
        }
    }

}

const std::unordered_set<point3D_t>& Map::GetModifiedPoint3DIds()
{
    return modified_point3D_ids_;
}


void Map::ClearModifiedPoint3DIds()
{
    modified_point3D_ids_.clear();
}



size_t Map::MergePoints3D(const std::unordered_set<point3D_t>& point3D_idxs,
                        const double& max_reproj_error)
{
    prepared_to_deleted_.clear();
    size_t num_merged = 0;
    for(const point3D_t& point3D_idx : point3D_idxs)
    {
        if(prepared_to_deleted_.count(point3D_idx) > 0)
            continue;

        num_merged += MergePoint3D(point3D_idx, max_reproj_error);
    }

    // 删除
    for(const point3D_t& point3D_idx : prepared_to_deleted_)
    {
        points3D_.erase(point3D_idx);
    }
    prepared_to_deleted_.clear();

    return num_merged;
}



bool Map::MergePoint3D(const point3D_t& point3D_idx,
                       const double& max_reproj_error)
{
    if(!HasPoint3DInMap(point3D_idx))
        return false;

    const Track& track = points3D_[point3D_idx].Track();

    for(const TrackElement& element : track.Elements())
    {
        const std::vector<SceneGraph::Correspondence> corrs =
                scene_graph_->FindCorrespondences(element.image_id, element.point2D_idx);

        for(const auto& corr : corrs)
        {
            if(registered_.count(corr.image_id) == 0)
                continue;
            if(!images_[corr.image_id].Point2DHasPoint3D(corr.point2D_idx))
                continue;
            point3D_t other_point3D_idx = images_[corr.image_id].GetPoint2D(corr.point2D_idx).Point3DId();

            if(point3D_idx == other_point3D_idx)
                continue;

            if(prepared_to_deleted_.count(other_point3D_idx) > 0)
                continue;
            // 合并
            bool is_merged = MergeTwoPoint3D(point3D_idx, other_point3D_idx, max_reproj_error);

            // 一旦合并成功， 那么就必须返回
            if(is_merged)
            {
                prepared_to_deleted_.insert(point3D_idx);
                prepared_to_deleted_.insert(other_point3D_idx);
                return true;
            }
        }
    }
    return false;
}

bool Map::MergeTwoPoint3D(const point3D_t& point3D_idx1,
                          const point3D_t& point3D_idx2,
                          const double& max_reproj_error)
{
    assert(prepared_to_deleted_.count(point3D_idx1) == 0);
    assert(prepared_to_deleted_.count(point3D_idx2) == 0);

    const Point3D& point3D1 = points3D_[point3D_idx1];
    const Point3D& point3D2 = points3D_[point3D_idx2];

    const cv::Vec3d& cv_point3D1 = point3D1.XYZ();
    const cv::Vec3d& cv_point3D2 = point3D2.XYZ();

    double w1 = point3D1.Track().Length();
    double w2 = point3D2.Track().Length();

    const cv::Vec3d& cv_merged_point3D = (w1 * cv_point3D1 + w2 * cv_point3D2) / (w1 + w2);

    size_t num_inliers = 0;
    double sum_error = 0;
    for(const Point3D& point3D : {point3D1, point3D2})
    {
        for(const TrackElement& element : point3D.Track().Elements())
        {
            const Image& image = images_[element.image_id];
            const Point2D& point2D = image.GetPoint2D(element.point2D_idx);

            bool has_positive_depth = Projection::HasPositiveDepth(cv_merged_point3D, image.Rotation(), image.Translation());

            if(!has_positive_depth)
            {
                break;
            }
            double error = Projection::CalculateReprojectionError(cv_merged_point3D, point2D.XY(), image.Rotation(), image.Translation(), K_);

            if(error > max_reproj_error)
            {
                break;
            }

            sum_error += error;
            num_inliers += 1;
        }
        if(num_inliers != point3D.Track().Length())
        {
            break;
        }
    }


    // 如果合并之后都是内点, 那么进行合并
    if(num_inliers == point3D1.Track().Length() + point3D2.Track().Length())
    {

        Track track;
        track.AddElements(point3D1.Track().Elements());
        track.AddElements(point3D2.Track().Elements());

        const cv::Vec3b& color1 = point3D1.Color();
        const cv::Vec3b& color2 = point3D2.Color();
        int sum_color_x = color1(0) + color2(0);
        int sum_color_y = color1(1) + color2(1);
        int sum_color_z = color1(2) + color2(2);

        int color_x = sum_color_x / 2;
        int color_y = sum_color_y / 2;
        int color_z = sum_color_z / 2;

        cv::Vec3b color(static_cast<uchar>(color_x),
                        static_cast<uchar>(color_y),
                        static_cast<uchar>(color_z));

        point3D_t point3D_idx = AddPoint3D(cv_merged_point3D, track, color, sum_error / num_inliers);
        MergePoint3D(point3D_idx, max_reproj_error);
        return true;
    }

    return false;
}


size_t Map::CompletePoints3D(const std::unordered_set<point3D_t>& point3D_idxs,
                           const double& max_reproj_error)
{
    size_t num_completed = 0;
    for(const point3D_t& point3D_idx : point3D_idxs)
    {
        num_completed += CompletePoint3D(point3D_idx, max_reproj_error);
    }
    return num_completed;
}


size_t Map::CompletePoints3DInImage(const std::unordered_set<image_t>& image_idxs,
                                    const double& max_reproj_error)
{
    std::unordered_set<point3D_t> point3D_idxs;

    for(const image_t& image_id : image_idxs)
    {
        const Image& image = images_[image_id];

        for(point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx)
        {
            if(image.GetPoint2D(point2D_idx).HasPoint3D())
            {
                point3D_t point3D_idx = image.GetPoint2D(point2D_idx).Point3DId();
                point3D_idxs.insert(point3D_idx);
            }
        }
    }
    return CompletePoints3D(point3D_idxs, max_reproj_error);
}

size_t Map::CompletePoint3D(const point3D_t& point3D_idx,
                          const double& max_reproj_error)
{
    if(!HasPoint3DInMap(point3D_idx))
        return 0;

    assert(points3D_.count(point3D_idx) != 0);

    size_t num_completed = 0;

    const Point3D& point3D = points3D_[point3D_idx];
    std::vector<TrackElement> queue;
    // 将同一个Track中的元素入队列，通过访问这些元素的匹配点，从而进行补全
    for(const TrackElement& element : point3D.Track().Elements())
    {
        queue.push_back(element);
    }

    const int kMaxTransitivity = 5;

    for(int transitivity = 0; transitivity < kMaxTransitivity; ++transitivity)
    {
        if(queue.empty())
            break;

        const auto& prev_queue = queue;
        queue.clear();

        for(const TrackElement& element : prev_queue)
        {
            // 得到该元素的匹配点
            const std::vector<SceneGraph::Correspondence> corrs =
                    scene_graph_->FindCorrespondences(element.image_id, element.point2D_idx);

            // 遍历这些匹配点
            for(const auto&  corr : corrs)
            {
                assert(element.image_id != corr.image_id);
                if(corr.image_id == element.image_id)
                    continue;
                // 如果匹配点所在的图片没有注册， 那么跳过
                if(registered_.count(corr.image_id) == 0)
                    continue;

                const Image& corr_image = images_[corr.image_id];
                const Point2D& corr_point2D = corr_image.GetPoint2D(corr.point2D_idx);

                // 如果匹配点已经有了3D点， 那么跳过
                if(corr_point2D.HasPoint3D())
                    continue;

                bool has_positive_depth = Projection::HasPositiveDepth(point3D.XYZ(), corr_image.Rotation(), corr_image.Translation());

                // 如果将3D点投影到匹配点所在的图像上， 没有正深度， 那么跳过
                if(!has_positive_depth)
                    continue;

                double error = Projection::CalculateReprojectionError(point3D.XYZ(), corr_point2D.XY(), corr_image.Rotation(), corr_image.Translation(), K_);

                // 如果3D点投影点与匹配点的距离过大，那么跳过
                if(error > max_reproj_error)
                    continue;

                AddObservation(point3D_idx, TrackElement(corr.image_id, corr.point2D_idx), error);

                // 递归补全
                queue.emplace_back(corr.image_id, corr.point2D_idx);
                num_completed += 1;
            }
        }
    }

    return num_completed;
}

size_t Map::FilterPoints3D(const std::unordered_set<point3D_t>& point3D_idxs,
                           const double& max_reproj_error,
                           const double& min_tri_angle)
{
    size_t num_filtered = 0;
    num_filtered += FilterPoints3DWithLargeReprojectionError(point3D_idxs, max_reproj_error);
    num_filtered += FilterPoints3DWithSmallTriangulationAngle(point3D_idxs, min_tri_angle);

    return num_filtered;
}
size_t Map::FilterPoints3DInImage(const std::unordered_set<image_t>& image_idxs,
                                  const double& max_reproj_error,
                                  const double& min_tri_angle)
{
    std::unordered_set<point3D_t> point3D_idxs;

    for(const image_t& image_id : image_idxs)
    {
        const Image& image = images_[image_id];

        for(point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx)
        {
            if(image.GetPoint2D(point2D_idx).HasPoint3D())
            {
                point3D_t point3D_idx = image.GetPoint2D(point2D_idx).Point3DId();
                point3D_idxs.insert(point3D_idx);
            }
        }
    }
    return FilterPoints3D(point3D_idxs, max_reproj_error, min_tri_angle);
}
size_t Map::FilterAllPoints3D(const double& max_reproj_error,
                              const double& min_tri_angle)
{
    std::unordered_set<point3D_t> point3D_idxs;
    for(const auto& ele : points3D_)
    {
        point3D_idxs.insert(ele.first);
    }
    return FilterPoints3D(point3D_idxs, max_reproj_error, min_tri_angle);
}

size_t Map::FilterPoints3DWithLargeReprojectionError(const std::unordered_set<point3D_t>& point3D_idxs,
                                                     const double& max_reproj_error)
{
    size_t num_filtered = 0;
    for(const point3D_t& point3D_idx : point3D_idxs)
    {
        if(!HasPoint3DInMap(point3D_idx))
            continue;
        Point3D& point3D = points3D_[point3D_idx];

        if(point3D.Track().Length() < 2)
        {
            RemovePoint3D(point3D_idx);
            continue;
        }
        double sum_reproj_error = 0;
        int sum_color_x = 0;
        int sum_color_y = 0;
        int sum_color_z = 0;

        std::vector<TrackElement> track_els_to_delete;
        const Track& track = point3D.Track();
        for(const TrackElement& element : track.Elements())
//        for(const TrackElement& element : point3D.Track().Elements()) // 这个循环语句不会进行遍历， 为什么？？
        {

            const Image& image = images_[element.image_id];
            const Point2D& point2D = image.GetPoint2D(element.point2D_idx);
            bool has_positive_depth = Projection::HasPositiveDepth(point3D.XYZ(), image.Rotation(), image.Translation());
            if(!has_positive_depth)
            {
                track_els_to_delete.push_back(element);
                continue;
            }
            double error = Projection::CalculateReprojectionError(point3D.XYZ(), point2D.XY(), image.Rotation(), image.Translation(), K_);

            if(error > max_reproj_error)
            {
                track_els_to_delete.push_back(element);
            }
            else
            {
                const cv::Vec3b& color = point2D.Color();
                sum_color_x += color(0);
                sum_color_y += color(1);
                sum_color_z += color(2);
                sum_reproj_error += error;
            }
        }

        if(track_els_to_delete.size() == point3D.Track().Length() ||
           track_els_to_delete.size() == point3D.Track().Length() - 1)
        {
            num_filtered += point3D.Track().Length();
            RemovePoint3D(point3D_idx);
        }
        else
        {
            for(const TrackElement& element : track_els_to_delete)
            {
                RemoveObservation(point3D_idx, element);
            }
            point3D.SetError(sum_reproj_error / point3D.Track().Length());
            cv::Vec3b color(static_cast<uchar>(sum_color_x / point3D.Track().Length()),
                            static_cast<uchar>(sum_color_y / point3D.Track().Length()),
                            static_cast<uchar>(sum_color_z / point3D.Track().Length()));
            point3D.SetColor(color);
        }
    }
    return num_filtered;
}
size_t Map::FilterPoints3DWithSmallTriangulationAngle(const std::unordered_set<point3D_t>& point3D_idxs,
                                                      const double& min_tri_angle)
{
    size_t num_filtered = 0;
    for(const point3D_t& point3D_idx : point3D_idxs)
    {
        if(!HasPoint3DInMap(point3D_idx))
            continue;

        const Point3D& point3D = points3D_[point3D_idx];

        if(point3D.Track().Length() < 2)
        {
            RemovePoint3D(point3D_idx);
            continue;
        }
        bool is_keep_point = false;
        for(size_t i = 0; i < point3D.Track().Length(); ++i)
        {
            const image_t& image_id1 = point3D.Track().Element(i).image_id;
            const Image& image1 = images_[image_id1];
            for(size_t j = 0; j < i; ++j)
            {
                const image_t& image_id2 = point3D.Track().Element(j).image_id;
                const Image& image2 = images_[image_id2];
                double tri_angle = Projection::CalculateParallaxAngle(point3D.XYZ(), image1.Rotation(), image1.Translation(), image2.Rotation(), image2.Translation());
                if(tri_angle >= min_tri_angle)
                {
                    is_keep_point = true;
                    break;
                }
            }
            if(is_keep_point)
                break;
        }
        if(!is_keep_point)
        {
            RemovePoint3D(point3D_idx);
            num_filtered += 1;
        }
    }
    return  num_filtered;
}

void Map::GetDataForVisualization(std::vector<cv::Point3f>& points3D,
                                  std::vector<cv::Vec3b>& colors,
                                  std::vector<cv::Mat>& Rs,
                                  std::vector<cv::Mat>& ts)
{
    GetPoint3DForVisualization(points3D, colors);
    GetCamerasForVisualization(Rs, ts);
}

void Map::GetPoint3DForVisualization(std::vector<cv::Point3f>& points3D,
                                     std::vector<cv::Vec3b>& colors)
{
    points3D.clear();
    colors.clear();

    points3D.reserve(points3D_.size());
    colors.reserve(points3D_.size());
    for(const auto& point3D_el : points3D_)
    {
        const cv::Vec3d& point3D = point3D_el.second.XYZ();
        const cv::Vec3b color = point3D_el.second.Color();

        points3D.push_back(cv::Point3f(static_cast<float>(point3D(0)),
                                       static_cast<float>(point3D(1)),
                                       static_cast<float>(point3D(2))));
        colors.push_back(color);
    }
}
void Map::GetCamerasForVisualization(std::vector<cv::Mat>& Rs,
                                     std::vector<cv::Mat>& ts)
{
    Rs.clear();
    ts.clear();

    Rs.reserve(registered_images_.size());
    ts.reserve(registered_images_.size());

    for(const image_t& registed_image_id : registered_images_)
    {
        const Image& registed_image = images_[registed_image_id];

        Rs.push_back(registed_image.Rotation().clone());
        ts.push_back(registed_image.Translation().clone());
    }
}

void Map::GetLocalBAData(BundleData& bundle_data)
{

    // 通过当前图像的特征点所在的Track（或通过当前图像的特征点的匹配关系）
    // 找到所有与当前图像有关系的图像
    // 遍历这些图像的Track， 生成优化关系

    // 最后一张图片， 是最新注册的图片
    const image_t& image_id = registered_images_[registered_images_.size() - 1];
    const Image& image = images_[image_id];
    std::unordered_map<image_t, size_t> num_covisible_of_images;

    for(point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx)
    {
        // 如果该2D点没有对应的3D点， 那么跳过
        const Point2D& point2D = image.GetPoint2D(point2D_idx);
        if(!point2D.HasPoint3D())
            continue;

        const point3D_t& point3D_idx = point2D.Point3DId();
        const Point3D& point3D = points3D_[point3D_idx];

        const Track& track = point3D.Track();

        for(const TrackElement& track_el : track.Elements())
        {
            // 与当前图像有关系的，已经注册了的图像
            // track中的图像， 当然是已经注册了的
            if(track_el.image_id == image_id)
                continue;

            num_covisible_of_images[track_el.image_id] += 1;
        }
    }

    const size_t kMaxRelatedImageNum = 5;


    std::vector<std::pair<image_t, size_t>> local_bundle;

    for(const auto& ele : num_covisible_of_images)
    {
        local_bundle.emplace_back(ele.first, ele.second);
    }

    const size_t num_related_images = std::min(kMaxRelatedImageNum, local_bundle.size());

    // 按照共视3D点的数量(从大到小),对图像进行排序
    std::partial_sort(local_bundle.begin(), local_bundle.begin() + num_related_images,
                      local_bundle.end(),
                      [](const std::pair<image_t, size_t>& image1,
                         const std::pair<image_t, size_t>& image2)
                      {
                            return image1.second > image2.second;
                      });

    std::vector<image_t> local_bundle_images_ids;
    std::unordered_set<image_t> local_bundle_images_vis;
    local_bundle_images_ids.push_back(image_id);
    local_bundle_images_vis.insert(image_id);

    std::cout << "local image ids : " << image_id << " ";
    for(size_t i = 0; i < num_related_images; ++i)
    {
        local_bundle_images_ids.push_back(local_bundle[i].first);
        local_bundle_images_vis.insert(local_bundle[i].first);
        std::cout << local_bundle[i].first << " ";
    }
    std::cout << std::endl;

    std::cout << "local bundle images size : " << local_bundle_images_ids.size() << std::endl;


    std::unordered_set<point3D_t> local_bundle_point3D_idxs;
    // 遍历相关图片的3D点, 这些3D点都是要进行优化的
    for(const image_t& local_bundle_image_id : local_bundle_images_ids)
    {
        const Image& local_bundle_image = images_[local_bundle_image_id];

        for(point2D_t point2D_idx = 0; point2D_idx < local_bundle_image.NumPoints2D(); ++point2D_idx)
        {
            // 如果该2D点没有对应的3D点， 那么跳过
            const Point2D& point2D = local_bundle_image.GetPoint2D(point2D_idx);
            if(!point2D.HasPoint3D())
                continue;

            const point3D_t& point3D_idx = point2D.Point3DId();
            local_bundle_point3D_idxs.insert(point3D_idx);
        }
    }



    std::unordered_map<point3D_t, BundleData::Landmark> landmarks;
    std::unordered_map<image_t, BundleData::CameraPose> camera_poses;
    std::unordered_set<image_t> constant_image_pose;

    for(const image_t& local_bundle_image_id : local_bundle_images_ids)
    {
        const cv::Mat& rotation = images_[local_bundle_image_id].Rotation();
        const cv::Mat& translation = images_[local_bundle_image_id].Translation();

        cv::Mat rvec;
        cv::Mat tvec;

        cv::Rodrigues(rotation, rvec);
        tvec = translation.clone();


        camera_poses[local_bundle_image_id] = BundleData::CameraPose(rvec, tvec);
    }


    constant_image_pose.insert(local_bundle_images_ids[local_bundle_images_ids.size() - 1]);

    for(const point3D_t& local_bundle_point3D_idx : local_bundle_point3D_idxs)
    {

        const Point3D& point3D = points3D_[local_bundle_point3D_idx];

        const Track& track = point3D.Track();



        std::vector<BundleData::Measurement> measurements;

        for(const TrackElement& element : track.Elements())
        {
            const image_t& image_id = element.image_id;

            // TOOD : 考虑可以将这里当做约束, 但是要考虑error的传递
            if(local_bundle_images_vis.count(image_id) == 0)
                continue;

            const point2D_t& point2D_idx = element.point2D_idx;

            const Point2D& point2D = images_[image_id].GetPoint2D(point2D_idx);

            measurements.emplace_back(image_id, point2D.XY());
        }

        landmarks[local_bundle_point3D_idx] = BundleData::Landmark(point3D.XYZ(), measurements);

    }

    bundle_data.K = K_;
    bundle_data.landmarks = landmarks;
    bundle_data.camera_poses = camera_poses;
    bundle_data.constant_camera_pose = constant_image_pose;

}
void Map::GetGlobalBAData(BundleData& bundle_data)
{


    std::unordered_map<point3D_t, BundleData::Landmark> landmarks;
    std::unordered_map<image_t, BundleData::CameraPose> camera_poses;
    std::unordered_set<image_t> constant_image_pose;

    for(const auto& registed_image_id : registered_images_)
    {
        const cv::Mat& rotation = images_[registed_image_id].Rotation();
        const cv::Mat& translation = images_[registed_image_id].Translation();

        cv::Mat rvec;
        cv::Mat tvec;

        cv::Rodrigues(rotation, rvec);
        tvec = translation.clone();

        camera_poses[registed_image_id] = BundleData::CameraPose(rvec, tvec);
    }

    constant_image_pose.insert(registered_images_[0]);


    for(const auto& point3D_el : points3D_)
    {
        const point3D_t& point3D_idx = point3D_el.first;
        const Point3D& point3D = point3D_el.second;

        const Track& track = point3D.Track();



        std::vector<BundleData::Measurement> measurements;

        for(const TrackElement& element : track.Elements())
        {
            const image_t& image_id = element.image_id;
            const point2D_t& point2D_idx = element.point2D_idx;

            const Point2D& point2D = images_[image_id].GetPoint2D(point2D_idx);

            measurements.emplace_back(image_id, point2D.XY());
        }

        landmarks[point3D_idx] = BundleData::Landmark(point3D.XYZ(), measurements);

    }


    bundle_data.K = K_;
    bundle_data.landmarks = landmarks;
    bundle_data.camera_poses = camera_poses;
    bundle_data.constant_camera_pose = constant_image_pose;


}

void Map::UpdateFromBAData(const BundleData& bundle_data)
{
    // update camera pose
    for(const auto& camera_psoe_el : bundle_data.camera_poses)
    {
        const image_t& image_id = camera_psoe_el.first;
        const BundleData::CameraPose& camera_pose = camera_psoe_el.second;

        cv::Mat R;
        cv::Mat t;
        cv::Rodrigues(camera_pose.rvec, R);
        t = camera_pose.tvec;


        images_[image_id].SetRotation(R);
        images_[image_id].SetTranslation(t);
    }

    // update point3D
    for(const auto& landmark_el : bundle_data.landmarks)
    {
        const point3D_t& point3D_idx = landmark_el.first;
        const BundleData::Landmark landmark = landmark_el.second;


        // 由于相机位姿和3D点变化了， 所以要对track重新计算误差
        double error = ComputeTrackError(landmark.point3D, points3D_[point3D_idx].Track());

        points3D_[point3D_idx].SetXYZ(landmark.point3D);
        points3D_[point3D_idx].SetError(error);
    }
}



struct Map::Statistics Map::Statistics()
{
    struct Statistics statistics;

    size_t num_observations = 0;
    size_t min_observations = (1 << 30);
    size_t max_observations = 0;


    // num observation
    for(const auto& image_el : images_)
    {
        const Image& image = image_el.second;
        // 累计， 已经注册的
        if(registered_.count(image_el.first) > 0)
        {
            size_t num_point3D = image.NumPoints3D();

            num_observations += num_point3D;
            min_observations = std::min(min_observations, num_point3D);
            max_observations = std::max(max_observations, num_point3D);
        }
    }

    double mean_observation_per_reg_image =  static_cast<double>(num_observations) / registered_.size();;



    double sum_reproj_error = 0;
    double min_reproj_error = (1e8);
    double max_reproj_erorr = 0;

    double sum_track_length = 0;
    size_t min_track_length = (1 << 30);
    size_t max_track_length = 0;

    for(const auto& point3D_el : points3D_)
    {
        const Point3D& point3D = point3D_el.second;

        // track length
        size_t track_length = point3D.Track().Length();
        sum_track_length += track_length;
        min_track_length = std::min(min_track_length, track_length);
        max_track_length = std::max(max_track_length, track_length);

        // reproj error
        assert(point3D.HasError());
        double error = point3D.Error();
        sum_reproj_error += error;
        min_reproj_error = std::min(min_reproj_error, error);
        max_reproj_erorr = std::max(max_reproj_erorr, error);
        // 检查这些误差是不是正确的
        // for debug
#ifdef DEBUG
        const Track& track = point3D.Track();
        double real_error = ComputeTrackError(point3D.XYZ(), track);
        if(std::fabs(real_error - point3D.Error()) >= 1e-6)
        {
            std::cout << "point3D_idx : " << point3D_el.first << ", real error : " << real_error << ", error : " << point3D.Error() << std::endl;

        }
        assert(std::fabs(real_error - point3D.Error()) < 1e-6);
#endif
    }

    double mean_track_length = sum_track_length / points3D_.size();
    double mean_reproj_error = sum_reproj_error / points3D_.size();


    statistics.num_points3D = points3D_.size();
    statistics.min_observations = min_observations;
    statistics.num_observations = num_observations;
    statistics.max_observations = max_observations;
    statistics.mean_observations_per_reg_image = mean_observation_per_reg_image;

    statistics.min_track_length = min_track_length;
    statistics.mean_track_length = mean_track_length;
    statistics.max_track_legnth = max_track_length;

    statistics.min_reporj_error = min_reproj_error;
    statistics.mean_reproj_error = mean_reproj_error;
    statistics.max_reproj_error = max_reproj_erorr;


    return statistics;
}

void Map::PrintStatistics(const struct Statistics& statistics)
{
    const size_t width = 35;
    std::cout.flags(std::ios::left); //左对齐
    std::cout << std::endl;
    std::cout << "--------------- Map Summary Start ---------------" << std::endl;
    std::cout << std::setw(width) << "Num points3D"                     << " : " << statistics.num_points3D << std::endl;
    std::cout << std::setw(width) << "min observations"                 << " : " << statistics.min_observations << std::endl;
    std::cout << std::setw(width) << "Num observations"                 << " : " << statistics.num_observations << std::endl;
    std::cout << std::setw(width) << "max observations"                 << " : " << statistics.max_observations << std::endl;
    std::cout << std::setw(width) << "Mean observations per reg image"  << " : " << statistics.mean_observations_per_reg_image << std::endl;

    std::cout << std::setw(width) << "Min track length"                 << " : " << statistics.min_track_length << std::endl;
    std::cout << std::setw(width) << "Mean track length"                << " : " << statistics.mean_track_length << std::endl;
    std::cout << std::setw(width) << "Max track length"                 << " : " << statistics.max_track_legnth << std::endl;

    std::cout << std::setw(width) << "Min reproj error"                 << " : " << statistics.min_reporj_error << std::endl;
    std::cout << std::setw(width) << "Mean reproj error"                << " : " << statistics.mean_reproj_error << std::endl;
    std::cout << std::setw(width) << "Max reproj error"                 << " : " << statistics.max_reproj_error << std::endl;
    std::cout << "--------------- Map Summary End ---------------"   << std::endl;
    std::cout << std::endl;
}



void Map::WriteOpenMVS(const std::string& directory)
{
    mkdir(directory.c_str(), S_IRWXU);

    MVS::Interface scene;

    // camera
    MVS::Interface::Platform platform;
    MVS::Interface::Platform::Camera camera;
    camera.K = K_;
    cv::Mat cv_image = cv::imread(images_[registered_images_[0]].ImageName());
    camera.width = cv_image.cols;
    camera.height = cv_image.rows;

    camera.R = MVS::Interface::Mat33d::eye();
    camera.C = MVS::Interface::Pos3d(0, 0, 0);


    platform.cameras.push_back(camera);


    // images & pose
    size_t nPoses = 0;
    size_t num_images = scene_graph_->NumImages();


    cv::Mat map1, map2;

    for(image_t image_id  = 0; image_id < num_images; ++image_id)
    {

        const Image& image = images_[image_id];
        const std::string src_image_path = image.ImageName();
        std::string src_image_directory;
        std::string src_image_name;
        Utils::SplitPath(src_image_path, src_image_directory, src_image_name);

        std::string dest_image_directory = Utils::UnionPath(directory, "undistorted_images");


        mkdir(dest_image_directory.c_str(), S_IRWXU);
        std::string dest_image_path = Utils::UnionPath(dest_image_directory, src_image_name);

        std::cout << "Undistort image # " << image_id + 1 << "/" << num_images << std::endl;
        cv::Mat cv_image = cv::imread(src_image_path);

        MVS::Interface::Image mvs_image;
        // TODO : 这里是相对路径, 要改成绝对路径
        std::cout << dest_image_path << std::endl;
        mvs_image.name = dest_image_path;
        mvs_image.platformID = 0;
        mvs_image.cameraID = 0;

        if(registered_.count(image_id) > 0)
        {
            cv::Mat cv_undistort_image = cv_image.clone();
            if(dist_coef_.at<double>(0, 0) != 0)
            {
                if(map1.size().area() == 0)
                {
                    cv::initUndistortRectifyMap(K_, dist_coef_, cv::Mat(), cv::Mat(), cv_image.size(), CV_32F, map1, map2);

                }
                cv::remap(cv_image, cv_undistort_image, map1, map2, cv::INTER_LINEAR);

            }
            cv_image = cv_undistort_image;

            MVS::Interface::Platform::Pose pose;
            mvs_image.poseID = platform.poses.size();
            pose.R = image.Rotation();
			cv::Mat C = -image.Rotation().t() * image.Translation();
            pose.C = MVS::Interface::Pos3d(C.at<double>(0, 0), C.at<double>(1, 0), C.at<double>(2, 0));

            platform.poses.push_back(pose);
            ++nPoses;
        }
        else
        {
            mvs_image.poseID = NO_ID;
        }

        cv::imwrite(dest_image_path, cv_image);
        scene.images.push_back(mvs_image);
    }

    scene.platforms.push_back(platform);



    // points3D
    for(const auto& point3D_el : points3D_)
    {
        const Point3D& point3D = point3D_el.second;
        const Track& track = point3D.Track();

        MVS::Interface::Vertex vert;
        MVS::Interface::Vertex::ViewArr& views = vert.views;


        for(const TrackElement& element : track.Elements())
        {
            MVS::Interface::Vertex::View view;
            view.imageID = element.image_id;
            view.confidence = 0;
            views.push_back(view);
        }

        if(views.size() < 2)
            continue;

        std::sort(views.begin(), views.end(),
                  [](const MVS::Interface::Vertex::View& view0, const MVS::Interface::Vertex::View& view1)
                    {
                        return view0.imageID < view1.imageID;
                    });
        const cv::Vec3d xyz = point3D.XYZ();
        vert.X = MVS::Interface::Pos3f(xyz(0), xyz(1), xyz(2));
        scene.vertices.push_back(vert);
    }

    // write OpenMVS data
    if (!MVS::ARCHIVE::SerializeSave(scene, Utils::UnionPath(directory, "scene.mvs")))
      return;

    std::cout
      << "Scene saved to OpenMVS interface format:\n"
      << " #platforms: " << scene.platforms.size() << std::endl;
      for (int i = 0; i < scene.platforms.size(); ++i)
      {
        std::cout << "  platform ( " << i << " ) #cameras: " << scene.platforms[i].cameras.size() << std::endl;
      }
      std::cout << " platform pose : " << scene.platforms[0].poses.size() << std::endl;
    std::cout
      << "  " << scene.images.size() << " images (" << nPoses << " calibrated)\n"
      << "  " << scene.vertices.size() << " Landmarks\n";

}

void Map::WritePLY(const std::string& path)
{
    // PLY data format
    // https://people.sc.fsu.edu/~jburkardt/data/ply/ply.html
    std::ofstream file;
    // trunc : 如果文件已存在则先删除该文件
    file.open(path.c_str(), std::ios::trunc);

    file << "ply" << std::endl;
    file << "format ascii 1.0" << std::endl;
    file << "element vertex " << points3D_.size() << std::endl;
    file << "property float x" << std::endl;
    file << "property float y" << std::endl;
    file << "property float z" << std::endl;
    file << "property uchar red" << std::endl;
    file << "property uchar green" << std::endl;
    file << "property uchar blue" << std::endl;
    file << "end_header" << std::endl;

    for(const auto& point3D_el : points3D_)
    {
        const Point3D& point3D = point3D_el.second;
        const cv::Vec3d& xyz = point3D.XYZ();
        const cv::Vec3b& color = point3D.Color();

        file << xyz(0) << " ";
        file << xyz(1) << " ";
        file << xyz(2) << " ";
        file << static_cast<int>(color(0)) << " ";
        file << static_cast<int>(color(1)) << " ";
        file << static_cast<int>(color(2)) << std::endl;;
    }
    file << std::endl;
    file.close();
}

void Map::WritePLYBinary(const std::string& path)
{
    std::ofstream file;
    // trunc : 如果文件已存在则先删除该文件
    file.open(path.c_str(), std::ios::trunc);

    file << "ply" << std::endl;
    file << "format binary_little_endian 1.0" << std::endl;
    file << "element vertex " << points3D_.size() << std::endl;
    file << "property float x" << std::endl;
    file << "property float y" << std::endl;
    file << "property float z" << std::endl;
    file << "property uchar red" << std::endl;
    file << "property uchar green" << std::endl;
    file << "property uchar blue" << std::endl;
    file << "end_header" << std::endl;



    for(const auto& point3D_el : points3D_)
    {
        const Point3D& point3D = point3D_el.second;
        const cv::Vec3d& xyz = point3D.XYZ();
        const cv::Vec3b& color = point3D.Color();

        const cv::Vec3f xyz_3f(xyz(0), xyz(1), xyz(2));
        file.write((char*)xyz_3f.val, sizeof (float) * 3);
        file.write((char*)color.val, sizeof(uchar) * 3);
    }

    file.close();
}



void Map::Write(const std::string& path)
{
    WriteCamera(Utils::UnionPath(path, "camara.txt"));
    WriteImages(Utils::UnionPath(path, "images.txt"));
    WritePoints3D(Utils::UnionPath(path, "points3D.txt"));
}
void Map::WriteCamera(const std::string& path)
{
    std::ofstream file;
    file.open(path.c_str(), std::ios::trunc);

    file << "# fx, fy, cx, cy, k1, k2, p1, p2" << std::endl;

    const double& fx = K_.at<double>(0, 0);
    const double& fy = K_.at<double>(1, 1);
    const double& cx = K_.at<double>(0, 2);
    const double& cy = K_.at<double>(1, 2);

    const double& k1 = dist_coef_.at<double>(0, 0);
    const double& k2 = dist_coef_.at<double>(1, 0);
    const double& p1 = dist_coef_.at<double>(2, 0);
    const double& p2 = dist_coef_.at<double>(3, 0);


    // TOOD : 在map中，没有存储畸变参数

    file << fx << " ";
    file << fy << " ";
    file << cx << " ";
    file << cy << " ";

    file << k1 << " ";
    file << k2 << " ";
    file << p1 << " ";
    file << p2 << std::endl;


    file.close();

}
void Map::WriteImages(const std::string& path)
{
    std::ofstream file;
    file.open(path.c_str(), std::ios::trunc);

    file << "# Image list with two lines of data per image:" << std::endl;
    file << "#   IMAGE_ID, R(0, 0), R(0, 1), R(0, 2), "
            "R(1, 0),R(1, 1), R(1, 2), "
            "R(2, 0), R(2, 1) R(2, 2), "
            "TX, TY, TZ, "
            "NAME"
         << std::endl;
    file << "#   POINTS2D[] as (X, Y, POINT3D_ID)" << std::endl;
    file << "# Number of images: " << registered_images_.size() << std::endl;

    for(const image_t& image_id : registered_images_)
    {
        std::ostringstream line;
        std::string line_string;

        line << image_id << " ";

        const Image& image = images_[image_id];
        const cv::Mat& R = image.Rotation();
        const cv::Mat& t = image.Translation();

        for(int y = 0; y < R.rows; ++y)
        {
            for(int x = 0; x < R.cols; ++x)
            {
                line << R.at<double>(y, x) << " ";
            }
        }
        for(int y = 0; y < t.rows; ++y)
        {
            line << t.at<double>(y, 0) << " ";
        }

        line << image.ImageName();

        file << line.str() << std::endl;

        line.str("");
        line.clear();


        for(point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx)
        {
            const Point2D& point2D = image.GetPoint2D(point2D_idx);
            const cv::Vec2d xy = point2D.XY();
            line << static_cast<float>(xy(0)) << " ";
            line << static_cast<float>(xy(1)) << " ";

            if(point2D.HasPoint3D())
            {
                line << point2D.Point3DId() << " ";
            }
            else
            {
                line << -1 << " ";
            }
        }
        line_string = line.str();
        line_string = line_string.substr(0, line_string.size() - 1);
        file << line_string << std::endl;
    }
    file.close();
}
void Map::WritePoints3D(const std::string& path)
{
    std::ofstream file;
    file.open(path.c_str(), std::ios::trunc);

    file << "# 3D point list with one line of data per point:" << std::endl;
    file << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
            "TRACK[] as (IMAGE_ID, POINT2D_IDX)"
         << std::endl;
    file << "# Number of points: " << points3D_.size() << std::endl;


    for (const auto& point3D_el : points3D_)
    {
        const point3D_t point3D_idx = point3D_el.first;
        const Point3D& point3D = point3D_el.second;
        const cv::Vec3d& xyz = point3D.XYZ();
        const cv::Vec3b& color = point3D.Color();
        const double& error = point3D.Error();

        file << point3D_idx << " ";
        file << xyz(0) << " ";
        file << xyz(1) << " ";
        file << xyz(2) << " ";
        file << static_cast<int>(color(0)) << " ";
        file << static_cast<int>(color(1)) << " ";
        file << static_cast<int>(color(2)) << " ";
        file << error << " ";

        std::ostringstream line;

        const Track& track = point3D.Track();
        for (const auto& track_el : track.Elements())
        {
            line << track_el.image_id << " ";
            line << track_el.point2D_idx << " ";
        }

        std::string line_string = line.str();
        line_string = line_string.substr(0, line_string.size() - 1);

        file << line_string << std::endl;
    }

    file.close();

}

double Map::ComputeTrackError(const cv::Vec3d& point3D,
                              const Track& track)
{
    double sum_residual = 0;
    for(const TrackElement& track_el : track.Elements())
    {
        const Image& image = images_[track_el.image_id];
        const Point2D point2D = image.GetPoint2D(track_el.point2D_idx);
        double error = Projection::CalculateReprojectionError(point3D, point2D.XY(), image.Rotation(), image.Translation(), K_);
        sum_residual += error;
    }
    return sum_residual / track.Length();
}

cv::Vec3b Map::ComputeTrackColor(const Track& track)
{
    int sum_color_x = 0;
    int sum_color_y = 0;
    int sum_color_z = 0;
    for(const TrackElement& track_el : track.Elements())
    {
        const image_t& image_id = track_el.image_id;
        const Image& image = images_[image_id];
        const cv::Vec3b color = image.GetPoint2D(track_el.point2D_idx).Color();
        sum_color_x += color(0);
        sum_color_y += color(1);
        sum_color_z += color(2);
    }

    int color_x = sum_color_x / track.Length();
    int color_y = sum_color_y / track.Length();
    int color_z = sum_color_z / track.Length();

    return cv::Vec3b(static_cast<uchar>(color_x),
                     static_cast<uchar>(color_y),
                     static_cast<uchar>(color_z));
}



void Map::Debug()
{
    for(const auto& point3D_el : points3D_)
    {
        const point3D_t& point3D_idx = point3D_el.first;
        const Point3D& point3D = point3D_el.second;
        const Track& track = point3D.Track();

        for(const TrackElement& track_el : track.Elements())
        {
            const image_t& image_id = track_el.image_id;
            const point2D_t& point2D_idx = track_el.point2D_idx;

            const Image& image = images_[image_id];
            const Point2D& point2D = image.GetPoint2D(point2D_idx);

            assert(point2D.HasPoint3D());
            if(point2D.Point3DId() != point3D_idx)
            {
                std::cout << image_id << " " << point2D_idx << " " << point3D_idx <<  " " << point2D.Point3DId() << std::endl;
            }
            assert(point2D.Point3DId() == point3D_idx);
        }

        double error = ComputeTrackError(point3D.XYZ(), track);
        assert(std::fabs(error - point3D.Error()) < 1e-6);

    }
}


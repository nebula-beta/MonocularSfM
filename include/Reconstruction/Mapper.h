#ifndef __MAPPER_H__
#define __MAPPER_H__

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "Feature/FeatureUtils.h"

#include "Common/Types.h"
#include "Common/Timer.h"
#include "Database/Database.h"
#include "Reconstruction/SceneGraph.h"
#include "Reconstruction/Image.h"
#include "Reconstruction/MapPoint.h"
#include "Reconstruction/Triangulation.h"
#include "Reconstruction/Filter.h"

#include "Visualization/Visualization.h"

#include "BundleAdjustment/BAData.h"
#include "BundleAdjustment/CeresBA.h"

namespace MonocularSfM
{


class Mapper
{
public:
    struct Config
    {
        // 相机内参
        float fx;
        float fy;
        float cx;
        float cy;

        // 畸变参数, 默认参数无效
        float k1 = 0.0;
        float k2 = 0.0;
        float p1 = 0.0;
        float p2 = 0.0;


        float max_2d_reprojection_error = 3.0;
        float max_3d_reprojection_error = 3.0;
        float min_parallax_degree = 1.5;


        int min_num_matches = 15;
        int max_initialize_num_trials = 15;
        int max_register_next_num_trials = 20;

        int initialize_pose_fail_threshold = 100;
        int initialize_triangulation_fail_threshold = 80;

        int register_2D_3D_correspodences_threshold = 10;
        int register_triangulation_fail_threshold = 10;



        float merge_reprojection_error = 4.0;
        float complete_reprojection_error = 4.0;
        float filtered_reprojection_error = 2.0;



        float ba_images_ratio = 1.05;


    };

    Mapper(const std::string& database_path, const Mapper::Config& config);
    ~Mapper();
    void DoMapper();

    // TODO : 实现下面的方法
    void Summary();
    void SaveForPMVS(const std::string& path);
    void SaveForOpenMVS(const std::string& path);

private:
    void SetUp();
    void LoadImageData(const image_t& image_id);
    std::vector<image_t> FindFirstInitialImage() const;
    std::vector<image_t> FindSecondInitialImage(image_t image_id) const;
    bool Initialize(image_t image_id1, image_t image_id2);
    void BuildInitialMap(const image_t& image_id1,
                         const image_t& image_id2,
                         const std::vector<cv::Point3f>& point3ds,
                         const cv::Mat& inlier_mask,
                         const std::vector<cv::DMatch>& matches);

    void AddNewMapPoint(const MapPoint& map_point, Color color);

    void SetImagePoint2D3DCorrespondence(const image_t& image_id,
                                         const point2D_t& point2D_idx,
                                         const point3D_t& point3D_idx);


    void UpdateImageVisable3DNum(const MapPoint& map_point, const int& inc);



    void SetRegistered(Image image);
    bool IsRegistered(image_t image_id) const;




    std::vector<image_t> FindNextImages() const;
    bool RegisterNextImage(image_t image_id);

    void Get2D3DCorrespondence(const Image& image,
                               std::vector<cv::Point2f>& point2Ds,
                               std::vector<cv::Point3f>& point3Ds,
                               std::vector<point2D_t>& p2D_idx,
                               std::vector<point3D_t>& p3D_idx);

    void UpdateMapPoint(const image_t& image_id,
                        const point2D_t& point2D_idx,
                        const point3D_t& point3D_idx);


    void TriangulateImage(const image_t& image_id);



    void Retriangulate();
    void RetriangulateImage(const image_t& image_id);




    void FilterAllMapPoints();
    void FilterMapPointsInImages(const std::vector<image_t>& image_ids);
    void FilterMapPoints(std::vector<point3D_t>& point3D_ids);
    bool FilterMapPoint(point3D_t point3D_idx);
    void DeletedMapPoints();


    void MergeAllMapPoints();
    void MergeMapPoints(const std::vector<point3D_t>& point3D_ids);
    size_t MergeMapPoint(point3D_t point3D_idx);
    point3D_t MergePoint3D(point3D_t point3D_idx1, point3D_t point3D_idx2, cv::Point3f merged_point3D);


    void CompleteAllMapPoints();
    void CompleteMapPoints(const std::vector<point3D_t>& point3D_ids);
    size_t CompleteMapPoint(point3D_t point3D_idx);

    void CompleteImages(const std::vector<image_t>& image_ids);
    void CompleteImage(const image_t& image_id);



    //BA
    void GetLocalBAData(image_t image_id, BAData& ba_data);
    void GetLocalBAData2(image_t image_id, BAData& ba_data);
    void LocalBA(image_t image_id);

    void GetGlobalBAData(BAData& ba_data);
    void GlobalBA();
    void UpdateFromBAData(BAData& ba_data);


    void ClearDeletedMapPoints();

    bool HasMapPoint(const point3D_t& point3D_idx);

    bool IsNeedUndistortFeature();
    void UndistortFeature(image_t image_id);

    float GetMapPointError(const MapPoint& map_point);
    float GetAveMapPointsError();



private:
    std::string database_path_;
    cv::Ptr<Database> database_;
    cv::Ptr<SceneGraph> scene_graph_;
    Mapper::Config config_;
    cv::Mat K_;
    cv::Mat dist_coef_;



    std::unordered_map<point3D_t, MapPoint> map_points_;
    std::unordered_map<point3D_t, Color> map_points_color_;

    std::unordered_map<image_t, Image> images_;
    std::unordered_map<image_t, bool> registered_;
    std::unordered_map<image_t, size_t> num_trial_registrations_;


    // track merge, complete, filter, re-riangulate
    std::unordered_map<point3D_t, std::unordered_set<point3D_t>> merge_trials_;
    std::unordered_map<point3D_t, bool> is_deleted_;
    std::unordered_map<image_t, int> num_re_triangulate_;



    size_t num_registered_;
    size_t num_point3D_index_;

    size_t prev_num_register_images_;
    size_t num_global_ba_;

};


} // namespace MonocularSfM
#endif //__MAPPER_H__

#ifndef __MAP_H__
#define __MAP_H__

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <fstream>

#include "Common/Types.h"

#include "Reconstruction/Utils.h"
#include "Reconstruction/Image.h"
#include "Reconstruction/Point3D.h"
#include "Reconstruction/Track.h"
#include "Reconstruction/SceneGraph.h"

#include "Optimizer/BundleData.h"


namespace MonocularSfM
{

class Map
{
public:
    struct Statistics
    {
        size_t num_points3D = 0;

        size_t min_observations = 0;
        size_t num_observations = 0;
        size_t max_observations = 0;


        double mean_observations_per_reg_image = 0;

        size_t min_track_length = 0;
        double mean_track_length = 0;
        size_t max_track_legnth = 0;

        double min_reporj_error = 0;
        double mean_reproj_error = 0;
        double max_reproj_error = 0;

    };
    struct CorrData
    {
        image_t image_id;
        point2D_t point2D_idx;
        cv::Mat R;
        cv::Mat t;
        cv::Vec2d point2D;
    };

public:

    Map(cv::Ptr<SceneGraph> scene_graph, const cv::Mat& K, const cv::Mat& dist_coef = cv::Mat::zeros(4, 1, CV_64F));

    void Load(cv::Ptr<Database> database);

    void AddImagePose(const image_t& image_id,
                      const cv::Mat& R,
                      const cv::Mat& t);


    point3D_t AddPoint3D(const cv::Vec3d& xyz, const Track& track);
    point3D_t AddPoint3D(const cv::Vec3d& xyz, const Track& track,
                         const double&error);
    point3D_t AddPoint3D(const cv::Vec3d& xyz, const Track& track,
                         const cv::Vec3b& color, const double&error);

    void RemovePoint3D(const point3D_t& point3D_idx);


    void AddObservation(const point3D_t& point3D_idx, const TrackElement& track_el, const double& error);
    void RemoveObservation(const point3D_t& point3D_idx, const TrackElement& track_el);


    std::vector<image_t> GetNextImageIds();

    point2D_t NumPoints2DInImage(const image_t& image_id);
    bool HasPoint3DInMap(const point3D_t& point3D_idx);
    bool HasPoint3DInImage(const image_t& image_id, const point2D_t& point2D_idx);
    Point3D GetPoint3DInImage(const image_t& image_id, const point2D_t& point2D_idx);


    size_t NumRegisteredImage();



    void Get2D2DCorrespoindencesBetweenImages(const image_t& image_id1,
                                              const image_t& image_id2,
                                              std::vector<cv::Vec2d>& points2D1,
                                              std::vector<cv::Vec2d>& points2D2,
                                              std::vector<point2D_t>& point2D_idxs1,
                                              std::vector<point2D_t>& point2D_idxs2);

    void Get2D3DCorrespondences(const image_t& image_id,
                                std::vector<cv::Vec2d>& points2D,
                                std::vector<cv::Vec3d>& points3D,
                                std::vector<point2D_t>& point2D_idxs,
                                std::vector<point3D_t>& point3D_idxs);

    void Get2D2DCorrespondences(const image_t& image_id,
                                std::vector<std::vector<CorrData>>& points2D_corr_datas);


    const std::unordered_set<point3D_t>& GetModifiedPoint3DIds();
private:
    void ClearModifiedPoint3DIds();


public:
    size_t MergePoints3D(const std::unordered_set<point3D_t>& point3D_idxs,
                       const double& max_reproj_error);

private:
    bool MergePoint3D(const point3D_t& point3D_idx,
                      const double& max_reproj_error);

    bool MergeTwoPoint3D(const point3D_t& point3D_idx1,
                         const point3D_t& point3D_idx2,
                         const double& max_reproj_error);
public:

    size_t CompletePoints3D(const std::unordered_set<point3D_t>& point3D_idxs,
                          const double& max_reproj_error);
    size_t CompletePoints3DInImage(const std::unordered_set<image_t>& image_idxs,
                                 const double& max_reproj_error);

private:
    size_t CompletePoint3D(const point3D_t& point3D_idx,
                         const double& max_reproj_error);
public:


    size_t FilterPoints3D(const std::unordered_set<point3D_t>& point3D_idxs,
                          const double& max_reproj_error,
                          const double& min_tri_angle);
    size_t FilterPoints3DInImage(const std::unordered_set<image_t>& image_idxs,
                                 const double& max_reproj_error,
                                 const double& min_tri_angle);

    size_t FilterAllPoints3D(const double& max_reproj_error,
                             const double& min_tri_angle);
private:
    size_t FilterPoints3DWithLargeReprojectionError(const std::unordered_set<point3D_t>& point3D_idxs,
                                                    const double& max_reproj_error);
    size_t FilterPoints3DWithSmallTriangulationAngle(const std::unordered_set<point3D_t>& point3D_idxs,
                                                     const double& min_tri_angle);
public:


    void GetDataForVisualization(std::vector<cv::Point3f>& points3D,
                                 std::vector<cv::Vec3b>& colors,
                                 std::vector<cv::Mat>& Rs,
                                 std::vector<cv::Mat>& ts);

    void GetPoint3DForVisualization(std::vector<cv::Point3f>& points3D,
                                    std::vector<cv::Vec3b>& colors);

    void GetCamerasForVisualization(std::vector<cv::Mat>& Rs,
                                    std::vector<cv::Mat>& ts);



    void GetLocalBAData(BundleData& bundle_data);
    void GetGlobalBAData(BundleData& bundle_data);
    void UpdateFromBAData(const BundleData& bundle_data);


    struct Statistics Statistics();
    void PrintStatistics(const struct Statistics& statistics);


    // write data to text file
    void WriteOpenMVS(const std::string& directory);
    void WritePLY(const std::string& path);
    void WritePLYBinary(const std::string& path);
    void Write(const std::string& path);
    void WriteCamera(const std::string& path);
    void WriteImages(const std::string& path);
    void WritePoints3D(const std::string& path);


private:
    double ComputeTrackError(const cv::Vec3d& point3D,
                             const Track& track);

    cv::Vec3b ComputeTrackColor(const Track& track);


public:
    void Debug();
private:

    std::vector<image_t> registered_images_;

    std::unordered_set<image_t> registered_;

    std::unordered_map<image_t, Image> images_;
    std::unordered_map<point3D_t, Point3D> points3D_;
    std::unordered_set<point3D_t> modified_point3D_ids_;

    // for merge point3d
    std::unordered_set<point3D_t> prepared_to_deleted_;


    point3D_t num_point3D_idx_;


    cv::Ptr<SceneGraph> scene_graph_;
    cv::Mat K_;
    cv::Mat dist_coef_;
};


} //namespace MonocularSfM

#endif //__MAP_H__

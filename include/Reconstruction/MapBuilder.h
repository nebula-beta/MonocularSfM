#ifndef __MAP_BUILDER_H__
#define __MAP_BUILDER_H__

#include <string>


#include "Common/Types.h"
#include "Common/Timer.h"

#include "Reconstruction/SceneGraph.h"
#include "Reconstruction/RegisterGraph.h"
#include "Reconstruction/Initializer.h"
#include "Reconstruction/Registrant.h"
#include "Reconstruction/Triangulator.h"
#include "Reconstruction/Map.h"

#include "Visualization/Visualization.h"

#include "Optimizer/BundleData.h"
#include "Optimizer/CeresBundleOptimizer.h"


namespace MonocularSfM
{

class MapBuilder
{
public:
    struct Parameters
    {
        // 相机内参
        double fx;
        double fy;
        double cx;
        double cy;

        // 畸变参数, 默认参数无效
        double k1 = 0.0;
        double k2 = 0.0;
        double p1 = 0.0;
        double p2 = 0.0;

        Initializer::Parameters init_params;          // 初始化时，所需要用到的参数
        Registrant::Parameters regis_params;          // 根据2D-3D点对应计算相机位姿态时，所需要用到的参数
        Triangulator::Parameters tri_params;          // 注册下一张图片，三角测量时，所需要用到的参数
        CeresBundelOptimizer::Parameters ba_params;   // BA优化时，所需要用到的参数

        size_t min_num_matches = 10;                  // 数据库中匹配数大于该阈值的图像对才会被加载进scene graph
        size_t max_num_init_trials = 100;

        double complete_max_reproj_error = 4.0;       // 补全track时，最大的重投影误差
        double merge_max_reproj_error = 4.0;          // 合并track时，最大的重投影误差
        double filtered_max_reproj_error = 4.0;       // 过滤track时，最大的重投影误差
        double filtered_min_tri_angle = 1.5;          // 过滤track时，最小要满足的角度


        double global_ba_ratio = 1.07;                // 当图像增加了该比率时，才会进行global BA

        bool is_visualization = true;                // 是否开启重建时， 点云、相机的可视化



    };
    struct Statistics
    {
        // TOOD
    };

public:
    MapBuilder(const std::string& database_path, const MapBuilder::Parameters &params);


    ////////////////////////////////////////////////////////////////////////////////
    // 重建时，需要调用的函数
    // 调用SetUp() 设置重建时需要加载的数据
    // 调用DoBuild() 来进行重建
    // 调用Summary() 输出重建结果的统计信息
    ////////////////////////////////////////////////////////////////////////////////
    void SetUp();
    void DoBuild();
    void Summary();

    //
    ////////////////////////////////////////////////////////////////////////////////
    // 将重建结果（相机参数、图片参数、3D点）写到文件中
    ////////////////////////////////////////////////////////////////////////////////
    void WriteOpenMVS(const std::string& directory);
    void WritePLY(const std::string& path);
    void WritePLYBinary(const std::string& path);
    void Write(const std::string& path);
    void WriteCamera(const std::string& path);
    void WriteImages(const std::string& path);
    void WritePoints3D(const std::string& path);

private:

    ////////////////////////////////////////////////////////////////////////////////
    // 寻找用于初始化的图像对
    ////////////////////////////////////////////////////////////////////////////////
    std::vector<image_t> FindFirstInitialImage() const;
    std::vector<image_t> FindSecondInitialImage(image_t image_id) const;

    ////////////////////////////////////////////////////////////////////////////////
    // 尝试进行初始化，直至成功，或者得到限定的初始化次数
    ////////////////////////////////////////////////////////////////////////////////
    bool TryInitialize();

    ////////////////////////////////////////////////////////////////////////////////
    // 尝试注册下一张图片
    ////////////////////////////////////////////////////////////////////////////////
    bool TryRegisterNextImage(const image_t& image_id);
    size_t Triangulate(const std::vector<std::vector<Map::CorrData>>& points2D_corr_datas,
                       double& ave_residual);


    ////////////////////////////////////////////////////////////////////////////////
    // 如果进行Local BA， 所需要进行的操作
    ////////////////////////////////////////////////////////////////////////////////
    void LocalBA();
    void MergeTracks();
    void CompleteTracks();
    void FilterTracks();

    ////////////////////////////////////////////////////////////////////////////////
    // 如果进行Global BA， 所需要进行的操作
    ////////////////////////////////////////////////////////////////////////////////
    void GlobalBA();
    void FilterAllTracks();
    // TODO : 对图像（或图像对）中没有3D点的2D点进行重建三角测量
    void Retriangulate();



    std::string                     database_path_;
    Parameters                      params_;


    cv::Ptr<Initializer>            initailizer_;
    cv::Ptr<Registrant>             registrant_;
    cv::Ptr<Triangulator>           triangulator_;

    cv::Ptr<RegisterGraph>          register_graph_;
    cv::Ptr<SceneGraph>             scene_graph_;
    cv::Ptr<Map>                    map_;
    cv::Ptr<CeresBundelOptimizer>   bundle_optimizer_;
    cv::Ptr<AsyncVisualization>     async_visualization_;


    cv::Mat                         K_;
    cv::Mat                         dist_coef_;


    Timer                           timer_for_initialize_;
    Timer                           timer_for_register_;
    Timer                           timer_for_triangulate_;
    Timer                           timer_for_merge_;
    Timer                           timer_for_complete_;
    Timer                           timer_for_local_filter_;
    Timer                           timer_for_local_ba_;
    Timer                           timer_for_global_filter_;
    Timer                           timer_for_global_ba_;

    Timer                           timer_for_visualization_;

    Timer                           timer_for_debug_;

    Timer                           timer_for_total_;

};


} //namespace MonocularSfM

#endif //__MAP_BUILDER_H__

#include "Common/Types.h"
#include "Visualization/Visualization.h"
#include "Reconstruction/Mapper.h"

#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz/widgets.hpp>

#include <string>


using namespace MonocularSfM;
AsyncVisualization::AsyncVisualization()
{

}


void AsyncVisualization::RunVisualizationThread()
{
    thread_ = new std::thread(&AsyncVisualization::RunVisualizationOnly, this);
}

void AsyncVisualization::WaitForVisualizationThread()
{
    thread_->join();
    delete thread_;
}

void AsyncVisualization::RunVisualizationOnly()
{
    std::string point_cloud_name = "point_cloud";


    // 创建显示窗口
    cv::viz::Viz3d window("Point Cloud Visualization");
    window.setBackgroundColor(cv::viz::Color::black());

    // 添加坐标系
    window.showWidget("Corrdinate Widget", cv::viz::WCoordinateSystem());


    while(!window.wasStopped())
    {
        if(is_point_cloud_update_)
        {
            if(point_cloud_change_count_ != 0)
            {
                window.removeWidget(point_cloud_name);
            }

            cv::viz::WCloud cloud(point_cloud_, colors_);
            window.showWidget(point_cloud_name, cloud);

            point_cloud_change_count_ ++;
            is_point_cloud_update_ = false;
        }
        if(is_camera_update_)
        {
            char name[1000];
            for(size_t i = 0; i < camera_count_; ++i)
            {
                sprintf(name, "cam_position%d", i);
                window.removeWidget(name);
            }

            for(size_t i = 0; i < Rs_.size(); ++i)
            {
                cv::Vec2f fov(1, 1);

                sprintf(name, "cam_position%d", i);

                // 不知道opencv底层是如何使用[R | t]的
                // 不过需要将旋转平移矩阵[R | t] 取逆矩阵 [R^-1 | -R^T * t]
                ts_[i] = -Rs_[i].t() * ts_[i];
                Rs_[i] = Rs_[i].inv();

                cv::Affine3f pose(Rs_[i], ts_[i]);
                cv::viz::WCameraPosition cam_position(fov, 0.5, cv::viz::Color::green());
                window.showWidget(name, cam_position);
                window.setWidgetPose(name, pose);
            }
            camera_count_ = Rs_.size();
            is_camera_update_ = false;
        }
        window.spinOnce(0, true);
    }

}

void AsyncVisualization::ShowPointCloud(std::vector<cv::Point3f>& point_cloud,
                                        std::vector<cv::Vec3b>& colors)
{
    point_cloud_ = std::move(point_cloud);
    colors_ = std::move(colors);

    is_point_cloud_update_ = true;
}
void AsyncVisualization::ShowPointCloud(const std::unordered_map<image_t, Image>& images,
                                        const std::unordered_map<point3D_t, MapPoint> &map_points,
                                        std::unordered_map<point3D_t, Color>& map_points_color)
{
    std::vector<cv::Point3f> point3Ds;
    std::vector<cv::Vec3b> colors;
    point3Ds.reserve(map_points.size());
    colors.reserve(map_points.size());
    size_t m = 0;
    for(auto& ele : map_points)
    {
        point3D_t point3D_idx = ele.first;
        const MapPoint& map_point = ele.second;

        if(std::isnan(map_point.Point3D().x) ||
           std::isnan(map_point.Point3D().y) ||
           std::isnan(map_point.Point3D().z))
        {
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
            std::cout << "nananananannnananananannnananananannnananananannnananananannnananananannnananananannnananananann " << std::endl;
        }
        point3Ds.push_back(map_point.Point3D());

        assert(map_points_color.find(point3D_idx) != map_points_color.end());

        Color color = map_points_color[point3D_idx];
        if(color == Color::COLOR_ORIGIN)
        {
            image_t image_id = map_point.Element(0).image_id;
            point2D_t point2D_idx = map_point.Element(0).point2D_idx;

            colors.push_back(images.at(image_id).Color(point2D_idx));

        }
        else if(color == Color::COLOR_RED)
        {
            colors.push_back(cv::Vec3b(0, 0, 255));
        }
        else
        {
            colors.push_back(cv::Vec3b(0, 255, 0));
        }

    }
    if(point3Ds.size() == 0)
        return;
    ShowPointCloud(point3Ds, colors);

    for(auto& ele : map_points_color)
    {
        ele.second = Color::COLOR_ORIGIN;
    }

}

void AsyncVisualization::ShowCameras(std::vector<cv::Mat> &Rs,
                                     std::vector<cv::Mat> &ts)
{
    Rs_ = std::move(Rs);
    ts_ = std::move(ts);

    is_camera_update_ = true;
}

void AsyncVisualization::ShowCameras(const std::unordered_map<image_t, Image> &images,
                                     const std::unordered_map<image_t, bool> registered)
{
    std::vector<cv::Mat> Rs;
    std::vector<cv::Mat> ts;
    for(const auto& elem : images)
    {
        if(registered.count(elem.first) > 0 && registered.at(elem.first))
        {
            Rs.push_back(elem.second.R().clone());
            ts.push_back(elem.second.t().clone());
        }

    }

    ShowCameras(Rs, ts);
}





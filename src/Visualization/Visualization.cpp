#include "Common/Types.h"
#include "Visualization/Visualization.h"

#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz/widgets.hpp>

#include <string>


using namespace MonocularSfM;
AsyncVisualization::AsyncVisualization()
{
    window_ = cv::viz::Viz3d ("Point Cloud Visualization");
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
//    cv::viz::Viz3d window("Point Cloud Visualization");
    window_.setBackgroundColor(cv::viz::Color::black());

    // 添加坐标系
    window_.showWidget("Corrdinate Widget", cv::viz::WCoordinateSystem());


    while(!window_.wasStopped())
    {
        if(is_point_cloud_update_)
        {
            if(point_cloud_change_count_ != 0)
            {
                window_.removeWidget(point_cloud_name);
            }

            cv::viz::WCloud cloud(point_cloud_, colors_);
            window_.showWidget(point_cloud_name, cloud);

            point_cloud_change_count_ ++;
            is_point_cloud_update_ = false;
        }
        if(is_camera_update_)
        {
            char name[1000];
            for(size_t i = 0; i < camera_count_; ++i)
            {
                sprintf(name, "cam_position%d", i);
                window_.removeWidget(name);
            }

            for(size_t i = 0; i < Rs_.size(); ++i)
            {
                cv::Vec2f fov(1, 1);

                sprintf(name, "cam_position%d", i);

                // 不知道opencv底层是如何使用[R | t]的
                // 不过需要将旋转平移矩阵[R | t] 取逆矩阵 [R^-1 | -R^T * t]
                ts_[i] = -Rs_[i].t() * ts_[i];
                Rs_[i] = Rs_[i].inv();

                // 点的类型要和相机的类型一致
                Rs_[i].convertTo(Rs_[i], CV_32F);
                ts_[i].convertTo(ts_[i], CV_32F);

                cv::Affine3f pose(Rs_[i], ts_[i]);

                cv::viz::WCameraPosition cam_position;
                if(Rs_.size() > 2 && i == Rs_.size() - 1)
                {
                    cam_position = cv::viz::WCameraPosition(fov, 0.1, cv::viz::Color::red());
                }
                else if(Rs_.size() > 2 && i == Rs_.size() - 2)
                {
                    // orange,  r = 255, g = 165, b = 0
                    int r = 255;
                    int g = 165;
                    int b = 0;
                    cam_position = cv::viz::WCameraPosition(fov, 0.1, cv::viz::Color(b, g, r));
                }
                else
                {
                    cam_position = cv::viz::WCameraPosition(fov, 0.1, cv::viz::Color::green());

                }
                window_.showWidget(name, cam_position);
                window_.setWidgetPose(name, pose);
            }
            camera_count_ = Rs_.size();
            is_camera_update_ = false;
        }
        window_.spinOnce(0, true);
    }
}

void AsyncVisualization::ShowPointCloud(std::vector<cv::Point3f>& point_cloud,
                                        std::vector<cv::Vec3b>& colors)
{
    point_cloud_ = std::move(point_cloud);
    colors_ = std::move(colors);

    is_point_cloud_update_ = true;
}

void AsyncVisualization::ShowCameras(std::vector<cv::Mat> &Rs,
                                     std::vector<cv::Mat> &ts)
{
    Rs_ = std::move(Rs);
    ts_ = std::move(ts);

    is_camera_update_ = true;
}


void AsyncVisualization::Close()
{
    window_.close();
}


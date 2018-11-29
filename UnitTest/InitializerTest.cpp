#include "Database/Database.h"
#include "Reconstruction/Initializer.h"
#include "Reconstruction/Utils.h"
#include "Visualization/Visualization.h"

#include <opencv2/opencv.hpp>
#include <string>

using namespace MonocularSfM;

void GetAlignedPointsFromMatches(const std::vector<cv::Point2f>& pts1,
                                              const std::vector<cv::Point2f>& pts2,
                                              const std::vector<cv::DMatch>& matches,
                                              std::vector<cv::Point2f>& aligned_pts1,
                                              std::vector<cv::Point2f>& aligned_pts2)
{
    aligned_pts1.resize(matches.size());
    aligned_pts2.resize(matches.size());
    for(size_t i = 0; i < matches.size(); ++i)
    {
        int queryIdx = matches[i].queryIdx;
        int trainIdx = matches[i].trainIdx;

        aligned_pts1[i] = pts1[queryIdx];
        aligned_pts2[i] = pts2[trainIdx];
    }
}


int main()
{


    std::string database_path = "/Users/anton/gerrard-hall.db";

    Database database ;
    database.Open(database_path);
    int image_id1 = 0;
    int image_id2 = 2;
    std::vector<cv::KeyPoint> kpts1 = database.ReadKeyPoints(image_id1);
    std::vector<cv::KeyPoint> kpts2 = database.ReadKeyPoints(image_id2);

    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;

    cv::KeyPoint::convert(kpts1, pts1);
    cv::KeyPoint::convert(kpts2, pts2);
    std::vector<cv::DMatch> matches = database.ReadMatches(image_id1, image_id2);

    std::vector<cv::Point2f> aligned_pts1;
    std::vector<cv::Point2f> aligned_pts2;
    GetAlignedPointsFromMatches(pts1, pts2, matches, aligned_pts1, aligned_pts2);



    cv::Mat K =  (cv::Mat_<double>(3, 3) << 3838.27, 0, 2808,
                          0, 3838.27, 1872,
                          0, 0, 1);

    Initializer::Parameters params;
    Initializer initializer(params, K);

    Initializer::Statistics statistics =  initializer.Initialize(Utils::Point2fToVector2d(aligned_pts1), Utils::Point2fToVector2d(aligned_pts2));

//    if(statistics.is_succeed)
//    {
//        AsyncVisualization async_visualization;
//        async_visualization.RunVisualizationThread();

//        std::vector<cv::Mat> Rs{statistics.R1, statistics.R2};
//        std::vector<cv::Mat> ts{statistics.t1, statistics.t2};

//        std::vector<cv::Point3f> points3D;
//        std::vector<cv::Vec3b> colors;
//        for(size_t i = 0; i < statistics.inlier_mask.size(); ++i)
//        {
//            if(!statistics.inlier_mask[i])
//                continue;
//            double x = statistics.points3D[i](0);
//            double y = statistics.points3D[i](1);
//            double z = statistics.points3D[i](2);
//            points3D.push_back(cv::Point3f(x, y, z));
//            colors.push_back(cv::Vec3b(0, 255, 0));

//        }
//        async_visualization.ShowPointCloud(points3D, colors);
//        async_visualization.ShowCameras(Rs, ts);

//        async_visualization.WaitForVisualizationThread();
//    }

    return 0;
}

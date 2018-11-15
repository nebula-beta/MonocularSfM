#include "BundleAdjustment/CeresBA.h"

#include "Reconstruction/Filter.h"
using namespace MonocularSfM;



void initLogging()
{
    google::InitGoogleLogging("SFM Bundle Adjustment!!!");
}


void CeresBA::Adjust(BAData &ba_data)
{
    assert(ba_data.K.type() == CV_32F);


    ceres::Problem problem;
    typedef cv::Matx<double, 1, 6> CameraVector;

    std::unordered_map<image_t, CameraVector> cameraPose6d;
    double focal[2] = {ba_data.K.at<float>(0, 0), ba_data.K.at<float>(1, 1)};

    for(const auto& ele : ba_data.camera_poses)
    {

        image_t image_id = ele.first;
        const CameraPose& camera_pose = ele.second;

        double angleAxis[3];
        cv::Mat rvec;
        cv::Rodrigues(camera_pose.R, rvec);
        assert(rvec.type() == CV_32F);

        angleAxis[0] = rvec.at<float>(0, 0);
        angleAxis[1] = rvec.at<float>(1, 0);
        angleAxis[2] = rvec.at<float>(2, 0);

        cameraPose6d[image_id] = CameraVector(
                                    angleAxis[0],
                                    angleAxis[1],
                                    angleAxis[2],
                                    camera_pose.t.at<float>(0, 0),
                                    camera_pose.t.at<float>(1, 0),
                                    camera_pose.t.at<float>(2, 0));

    }


    std::vector<cv::Vec3d> point3Ds(ba_data.landmarks.size());
    std::vector<point3D_t> point3D_idxs;

    int m = 0;
    for(const Landmark& landmark : ba_data.landmarks)
    {
        point3D_idxs.push_back(landmark.point3D_idx);
        cv::Point3f point3D = landmark.point3D;

        point3Ds[m] = cv::Vec3d(point3D.x, point3D.y, point3D.z);

        for(const Measurement& meas : landmark.measurements)
        {
            image_t image_id = meas.image_id;
            cv::Point2f point2D = meas.measurement;

            point2D.x -= ba_data.K.at<float>(0, 2);
            point2D.y -= ba_data.K.at<float>(1, 2);
            ceres::CostFunction* cost_function = SimpleReprojectError::Create(point2D.x, point2D.y);
            problem.AddResidualBlock(cost_function, NULL, cameraPose6d[image_id].val, point3Ds[m].val, focal);

        }
        m+=1;

    }

    problem.SetParameterBlockConstant(focal);

    ceres::Solver::Options options;
    const size_t kMaxNumImagesDirectDenseSolver = 50;
    const size_t kMaxNumImagesDirectSparseSolver = 1000;
    if (ba_data.camera_poses.size() <= kMaxNumImagesDirectDenseSolver)
    {
      options.linear_solver_type = ceres::DENSE_SCHUR;
    }
    else if (ba_data.camera_poses.size() <= kMaxNumImagesDirectSparseSolver)
    {
      options.linear_solver_type = ceres::SPARSE_SCHUR;
    }


//    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;


    const size_t kMinNumRegImages = 10;

    if(ba_data.camera_poses.size() < kMinNumRegImages)
    {
        options.function_tolerance /= 10;
        options.gradient_tolerance /= 10;
        options.parameter_tolerance /= 10;

        options.max_num_iterations *= 2;
        options.max_linear_solver_iterations = 200;
    }
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
//    std::cout << summary.FullReport() << std::endl;
    if(not (summary.termination_type == ceres::CONVERGENCE))
    {
        std::cout << "Bundle Adjustment failed." << std::endl;

        return;
    }
    else
    {
        std::cout << std::endl
        << "Bundle Adjustment statistics (approximated RMSE):\n"
        << " #residuals: " << summary.num_residuals << "\n"
        << " Initial RMSE: " << std::sqrt(summary.initial_cost * 2/ summary.num_residuals) << "\n"
        << " Final RMSE: " << std::sqrt(summary.final_cost * 2 / summary.num_residuals) << "\n"
        << " Time (s): " << summary.total_time_in_seconds << "\n"
        << std::endl;
    }


    // 更新
    for(const auto& ele : cameraPose6d)
    {
        image_t image_id = ele.first;
        CameraVector camera_pose = ele.second;
        cv::Mat rvec = (cv::Mat_<double>(3, 1) <<
                        camera_pose(0),
                        camera_pose(1),
                        camera_pose(2));
        cv::Rodrigues(rvec, rvec);

        rvec.convertTo(ba_data.camera_poses[image_id].R, CV_32F);


        ba_data.camera_poses[image_id].t.at<float>(0, 0) = camera_pose(3);
        ba_data.camera_poses[image_id].t.at<float>(1, 0) = camera_pose(4);
        ba_data.camera_poses[image_id].t.at<float>(2, 0) = camera_pose(5);
    }

    m = 0;
    for(Landmark& landmark : ba_data.landmarks)
    {
        assert(point3D_idxs[m] == landmark.point3D_idx);
        landmark.point3D.x = point3Ds[m][0];
        landmark.point3D.y = point3Ds[m][1];
        landmark.point3D.z = point3Ds[m][2];
        m += 1;


        ///  下面的语句用来DEBUG， 判断BA是否成功
//        for(const Measurement& meas : landmark.measurements)
//        {
//            cv::Point2f pt = meas.measurement;
//            float error = PointFilter::CalculateReprojectionError(landmark.point3D, pt,
//                                                                  ba_data.camera_poses[meas.image_id].R,
//                                                                  ba_data.camera_poses[meas.image_id].t,
//                                                                  ba_data.K);
//            std::cout << error << std::endl;
//        }

    }

}

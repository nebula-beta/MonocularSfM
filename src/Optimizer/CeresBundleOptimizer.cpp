#include "Optimizer/CeresBundleOptimizer.h"


#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/loss_function.h>

using namespace MonocularSfM;


void initLogging()
{
    google::InitGoogleLogging("SFM Bundle Adjustment!!!");
}



////////////////////////////////////////////////////////////////////////////////
// cost function
////////////////////////////////////////////////////////////////////////////////
struct BundleAutoDiffConstantFocalCostFunction
{
    BundleAutoDiffConstantFocalCostFunction(double observed_x,
                                            double observed_y,
                                            double focal_x,
                                            double focal_y) :
        observed_x(observed_x), observed_y(observed_y), focal_x(focal_x), focal_y(focal_y){}

    template<typename T>
    bool operator()(const T* const rvec,
                    const T* const tvec,
                    const T* const point,
                          T* residuals) const
    {
        T p[3];
        //对点point施加rvec所对应的旋转，　结果存储在p中
        ceres::AngleAxisRotatePoint(rvec, point, p);
        //加上平移
        p[0] += tvec[0];
        p[1] += tvec[1];
        p[2] += tvec[2];

        //齐次坐标归一化
        const T xp = p[0] / p[2];
        const T yp = p[1] / p[2];

        const T predicted_x = T(focal_x) * xp;
        const T predicted_y = T(focal_y) * yp;

        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }


    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y,
                                       const double focal_x,
                                       const double focal_y)
    {
        //2 表示残差项的维度为2
        //3	表示rvec的维度为3
        //3	表示tvec的维度为3
        //3 表示point的维度为3
        return (new ceres::AutoDiffCostFunction<BundleAutoDiffConstantFocalCostFunction, 2, 3, 3, 3>(
                new BundleAutoDiffConstantFocalCostFunction(observed_x, observed_y, focal_x, focal_y)));
    }
    double observed_x;
    double observed_y;
    double focal_x;
    double focal_y;
};



struct BundleAutoDiffCostFunction
{
    BundleAutoDiffCostFunction(double observed_x, double observed_y) :
        observed_x(observed_x), observed_y(observed_y){}

    template<typename T>
    bool operator()(const T* const rvec,
                    const T* const tvec,
                    const T* const point,
                    const T* const focal,
                          T* residuals) const
    {
        T p[3];
        //对点point施加rvec所对应的旋转，　结果存储在p中
        ceres::AngleAxisRotatePoint(rvec, point, p);
        //加上平移
        p[0] += tvec[0];
        p[1] += tvec[1];
        p[2] += tvec[2];

        //齐次坐标归一化
        const T xp = p[0] / p[2];
        const T yp = p[1] / p[2];

        const T predicted_x = focal[0] * xp;
        const T predicted_y = focal[1] * yp;

        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }


    static ceres::CostFunction* Create(const double observed_x, const double observed_y)
    {
        //2 表示残差项的维度为2
        //3	表示rvec的维度为3
        //3	表示tvec的维度为3
        //3 表示point的维度为3
        //2 表示focal的维度为2
        return (new ceres::AutoDiffCostFunction<BundleAutoDiffCostFunction, 2, 3, 3, 3, 2>(
                new BundleAutoDiffCostFunction(observed_x, observed_y)));
    }
    double observed_x;
    double observed_y;
};



/// 使用数值求导， 用来DEBUG
struct BundleNumericDiffCostFunction
{
    BundleNumericDiffCostFunction(double observed_x, double observed_y) :
        observed_x(observed_x), observed_y(observed_y){}

    bool operator()(const double* const rvec,
                    const double* const tvec,
                    const double* const point,
                    const double* const focal,
                          double* residuals) const
    {
        double p[3];
        //对点point施加camera[0,1,2]所对应的旋转，　结果存储在p中
        ceres::AngleAxisRotatePoint(rvec, point, p);
        //加上平移
        p[0] += tvec[0];
        p[1] += tvec[1];
        p[2] += tvec[2];

        //齐次坐标归一化
        const double xp = p[0] / p[2];
        const double yp = p[1] / p[2];

        const double predicted_x = focal[0] * xp;
        const double predicted_y = focal[1] * yp;

        residuals[0] = predicted_x - double(observed_x);
        residuals[1] = predicted_y - double(observed_y);
        std::cout << residuals[0] << " " << residuals[1] << std::endl;
        return true;
    }


    static ceres::CostFunction* Create(const double observed_x, const double observed_y)
    {
        //2 表示残差项的维度为2
        //6	表示camera的维度为6
        //3 表示point的维度为3
        //1 表示focal的维度为2
        return (new ceres::NumericDiffCostFunction<BundleNumericDiffCostFunction, ceres::CENTRAL, 2, 3, 3, 3, 2>(
                new BundleNumericDiffCostFunction(observed_x, observed_y)));
    }
    double observed_x;
    double observed_y;
};







////////////////////////////////////////////////////////////////////////////////
// Optimizer
////////////////////////////////////////////////////////////////////////////////

CeresBundelOptimizer::CeresBundelOptimizer(const Parameters& params) : params_(params)
{

}


bool CeresBundelOptimizer::Optimize(BundleData& bundle_data)
{
    assert(bundle_data.K.type() == CV_64F);


#ifdef DEBUG
    double before_error = bundle_data.Debug();
#endif

    double fx = bundle_data.K.at<double>(0, 0);
    double fy = bundle_data.K.at<double>(1, 1);
    double cx = bundle_data.K.at<double>(0, 2);
    double cy = bundle_data.K.at<double>(1, 2);

    double focal[2] = {fx, fy};



    ceres::Problem problem;


    ceres::LossFunction* loss_function = nullptr;
//    new ceres::CauchyLoss(params_.loss_function_scale);


    for(auto& landmark_el : bundle_data.landmarks)
    {
        point3D_t lankmark_id = landmark_el.first;
        BundleData::Landmark& landmark = landmark_el.second;
        for(auto& measurement : landmark.measurements)
        {
            const image_t& image_id = measurement.image_id;
            const cv::Vec2d& point2D = measurement.point2D;
            double u = point2D(0) - cx;
            double v = point2D(1) - cy;


            if(params_.refine_focal_length)
            {
                ceres::CostFunction* cost_function = BundleAutoDiffCostFunction::Create(u, v);

                problem.AddResidualBlock(cost_function, loss_function,
                                         reinterpret_cast<double*>(bundle_data.camera_poses[image_id].rvec.data),
                                         reinterpret_cast<double*>(bundle_data.camera_poses[image_id].tvec.data),
                                         landmark.point3D.val,
                                         focal);

            }
            else
            {
                ceres::CostFunction* cost_function = BundleAutoDiffConstantFocalCostFunction::Create(u, v, fx, fy);
                problem.AddResidualBlock(cost_function, loss_function,
                                         reinterpret_cast<double*>(bundle_data.camera_poses[image_id].rvec.data),
                                         reinterpret_cast<double*>(bundle_data.camera_poses[image_id].tvec.data),
                                         landmark.point3D.val);
            }


        }
    }


//    if(params_.refine_focal_length)
//    {
//        problem.SetParameterBlockConstant(focal);
//    }


    for(const image_t& const_image_id : bundle_data.constant_camera_pose)
    {
        problem.SetParameterBlockConstant(reinterpret_cast<double*>(bundle_data.camera_poses[const_image_id].rvec.data));
        problem.SetParameterBlockConstant(reinterpret_cast<double*>(bundle_data.camera_poses[const_image_id].tvec.data));
    }

    ceres::Solver::Options options;

    const size_t kMaxNumImagesDirectDenseSolver = 50;
    const size_t kMaxNumImagesDirectSparseSolver = 1000;
    if (bundle_data.camera_poses.size() <= kMaxNumImagesDirectDenseSolver)
    {
        options.linear_solver_type = ceres::DENSE_SCHUR;
    }
    else if (bundle_data.camera_poses.size() <= kMaxNumImagesDirectSparseSolver)
    {
         options.linear_solver_type = ceres::SPARSE_SCHUR;
    }

    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;


    const size_t kMinNumRegImages = 10;

    // 如果图片数量少， 那么就尽可能让其优化得更好些
    if(bundle_data.camera_poses.size() < kMinNumRegImages)
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

        return false;
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

    if(params_.refine_focal_length)
    {
        bundle_data.K.at<double>(0, 0) = focal[0];
        bundle_data.K.at<double>(1, 1) = focal[1];
    }


#ifdef DEBUG
    double after_error = bundle_data.Debug();
    std::cout << "Before Optimizer Error : " << before_error << std::endl;
    std::cout << "After Optimizer Error  : " << after_error << std::endl;
#endif

    return true;

}

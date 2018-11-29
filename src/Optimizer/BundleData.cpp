#include "Optimizer/BundleData.h"

#include "Reconstruction/Projection.h"

using namespace MonocularSfM;



double BundleData::Debug()
{
    double sum_residuals = 0;
    double num = 0;
    for(auto& landmark_el : landmarks)
    {
        const Landmark& landmark = landmark_el.second;
        const cv::Vec3d& point3D = landmark.point3D;

        double temp_sum_error = 0;
        for(const Measurement& meas : landmark.measurements)
        {
            const image_t& image_id = meas.image_id;
            const cv::Vec2d& point2D = meas.point2D;

            cv::Mat R;

            cv::Rodrigues(camera_poses[image_id].rvec, R);
            const cv::Mat& t = camera_poses[image_id].tvec;

            double error = Projection::CalculateReprojectionError(point3D, point2D, R, t, K);
            temp_sum_error += error;
        }
        sum_residuals += temp_sum_error / landmark.measurements.size();
        num += 1;
    }

    return sum_residuals / num;
}


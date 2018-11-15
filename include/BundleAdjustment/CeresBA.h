#ifndef __CERES_BA_H__
#define __CERES_BA_H__


#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/loss_function.h>


#include "BundleAdjustment/BAData.h"



void initLogging();
namespace MonocularSfM
{



class CeresBA
{
public:
    static void Adjust(BAData& ba_data);
};







struct SimpleReprojectError
{
    SimpleReprojectError(double observed_x, double observed_y) :
        observed_x(observed_x), observed_y(observed_y){}

    template<typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    const T* const focal,
                          T* residuals) const
    {
        T p[3];
        //对点point施加camera[0,1,2]所对应的旋转，　结果存储在p中
        ceres::AngleAxisRotatePoint(camera, point, p);
        //加上平移
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

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
        //6	表示camera的维度为6
        //3 表示point的维度为3
        //1 表示focal的维度为2
        return (new ceres::AutoDiffCostFunction<SimpleReprojectError, 2, 6, 3, 2>(
                new SimpleReprojectError(observed_x, observed_y)));
    }
    double observed_x;
    double observed_y;
};



/// 使用数值求导， 用来DEBUG
struct SimpleReprojectError2
{
    SimpleReprojectError2(double observed_x, double observed_y) :
        observed_x(observed_x), observed_y(observed_y){}

    bool operator()(const double* const camera,
                    const double* const point,
                    const double* const focal,
                          double* residuals) const
    {
        double p[3];
        //对点point施加camera[0,1,2]所对应的旋转，　结果存储在p中
        ceres::AngleAxisRotatePoint(camera, point, p);
        //加上平移
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

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
        return (new ceres::NumericDiffCostFunction<SimpleReprojectError2, ceres::CENTRAL, 2, 6, 3, 2>(
                new SimpleReprojectError2(observed_x, observed_y)));
    }
    double observed_x;
    double observed_y;
};




} // namspace MonocularSfM

#endif // __CERES_BA_H__

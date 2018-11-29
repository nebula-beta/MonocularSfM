#ifndef __CERES_BUNDLE_OPTIMIZER_H__
#define __CERES_BUNDLE_OPTIMIZER_H__


#include "Optimizer/BundleData.h"


void initLogging();

namespace MonocularSfM
{


class CeresBundelOptimizer
{
public:
    struct Parameters
    {
        int min_observation_per_image = 10;
        bool refine_focal_length = false;
        double loss_function_scale = 1.0;

    };

    struct Statistics
    {
        bool is_succeed = false;

    };

    CeresBundelOptimizer(const Parameters& params);
    bool Optimize(BundleData& bundle_data);


private:

    Parameters params_;


};

} // MonocularSfM


#endif //__CERES_BUNDLE_OPTIMIZER_H__

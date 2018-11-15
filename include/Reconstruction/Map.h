#ifndef __MAP_H__
#define __MAP_H__

#include <unordered_map>

#include "Common/Types.h"
#include "Reconstruction/SceneGraph.h"
#include "Reconstruction/Image.h"
#include "Reconstruction/MapPoint.h"

namespace MonocularSfM
{


enum
{
    COLOR_ORIGIN,
    COLOR_RED,
    COLOR_GREEN
};

class Map
{
public:
    Map() { }

    void BuildInitialMap(const Image& image1,
                         const Image& image2,
                         std::unordered_map<point3D_t, MapPoint>& init_map_points);
    void ResetMap();

    void Get2D3DCorrespondence(const Image& image,
                               std::vector<cv::Point2d>& point2Ds,
                               std::vector<cv::Point3f>& point3Ds);


    void Get2D2DCorrespondence(const Image& image,
                               std::vector<cv::Point2f> point2Ds1,
                               std::vector<cv::Point2f> point2Ds2);

    void UpdateMap(const Image& new_image, std::unordered_map<point3D_t, MapPoint> new_map_points);

    void UpdateNumObservation3D(image_t image_id, int inc);
    int GetNumObservation3D(image_t image_id);


private:





private:
    std::unordered_map<point3D_t, MapPoint> map_points_;
    std::unordered_map<point3D_t, int> map_points_status_;

    std::unordered_map<image_t, Image> images_;
    std::unordered_map<image_t, int> num_observation_point3D_;

};


} // namespace MonocularSfM


#endif // __MAP_H__

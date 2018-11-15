#include "Reconstruction/Map.h"

using namespace MonocularSfM ;


void Map::BuildInitialMap(const Image& image1,
                          const Image& image2,
                          std::unordered_map<point3D_t, MapPoint>& init_map_points)
{
    images_[image1.ImageId()] = image1;
    images_[image2.ImageId()] = image2;

    map_points_ = init_map_points;

    for(const auto& map_point : map_points_)
    {
        map_points_status_[map_point.first] = COLOR_ORIGIN;
    }

}

#include "Reconstruction/MapBuilder.h"
#include <string>
using namespace std;
using namespace MonocularSfM;



cv::String UnionImagePath2(const std::string& images_path, const cv::String& image_path)
{
    if(images_path[images_path.size() - 1] == '/')
    {
        return cv::String(images_path) + image_path;
    }
    else
    {
        return cv::String(images_path + "/") + image_path;
    }
}


int main(int argc, char** argv)
{

    MapBuilder::Parameters params;

    assert(argc == 7 || argc == 11);



    string database_path = argv[1];
    string output_path = "";

    // 相机内参
    params.fx = std::atof(argv[2]);
    params.fy = std::atof(argv[3]);
    params.cx = std::atof(argv[4]);
    params.cy = std::atof(argv[5]);

    if(argc == 7)
    {
        output_path = argv[6];
    }
    if(argc == 10)
    {
        // 畸变参数
        params.k1 = std::atof(argv[6]);
        params.k2 = std::atof(argv[7]);
        params.p1 = std::atof(argv[8]);
        params.p2 = std::atof(argv[9]);
        output_path = argv[10];
    }


    MapBuilder map_builder(database_path, params);

    map_builder.SetUp();
    map_builder.DoBuild();
    map_builder.WritePLY(UnionImagePath2(output_path, "points3D.ply"));
    map_builder.WritePLYBinary(UnionImagePath2(output_path, "points3D_binary.ply"));
    map_builder.Write(output_path);
}

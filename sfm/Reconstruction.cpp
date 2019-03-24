#include "Reconstruction/MapBuilder.h"
#include "Reconstruction/Utils.h"
#include <string>
using namespace std;
using namespace MonocularSfM;





int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cout << "You need specify the YAML file path!" << std::endl;
        exit(-1);
    }


    cv::FileStorage fs(argv[1], cv::FileStorage::READ);

    if(!fs.isOpened())
    {
        std::cout << "YAML file : " << argv[1] << " can't not open!" << std::endl;
        exit(-1);
    }


    MapBuilder::Parameters params;


    string image_path ;
    string database_path;


    fs["image_path"] >> image_path;
    fs["database_path"] >> database_path;

    // 相机内参
    fs["Reconstruction.Camera.fx"] >> params.fx;
    fs["Reconstruction.Camera.fy"] >> params.fy;
    fs["Reconstruction.Camera.cx"] >> params.cx;
    fs["Reconstruction.Camera.cy"] >> params.cy;

    fs["Reconstruction.Camera.k1"] >> params.k1;
    fs["Reconstruction.Camera.k2"] >> params.k2;
    fs["Reconstruction.Camera.p1"] >> params.p1;
    fs["Reconstruction.Camera.p2"] >> params.p2;




    string output_path;
    fs["Reconstruction.output_path"] >> output_path;
    fs["Reconstruction.is_visualization"] >> params.is_visualization;


    MapBuilder map_builder(image_path, database_path, params);

    map_builder.SetUp();
    map_builder.DoBuild();

    map_builder.WriteCOLMAP(output_path);
    map_builder.WriteOpenMVS(output_path);
    map_builder.WritePLY(Utils::UnionPath(output_path, "points3D.ply"));
    map_builder.WritePLYBinary(Utils::UnionPath(output_path, "points3D_binary.ply"));

}

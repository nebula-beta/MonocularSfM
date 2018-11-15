
add_library(
    reconstruction
    STATIC
    ../../include/Reconstruction/SceneGraph.h               SceneGraph.cpp
    ../../include/Reconstruction/Image.h                    Image.cpp
    ../../include/Reconstruction/MapPoint.h                 MapPoint.cpp
    ../../include/Reconstruction/Map.h                      Map.cpp
    ../../include/Reconstruction/Triangulation.h            Triangulation.cpp
    ../../include/Reconstruction/Filter.h                   Filter.cpp

    ../../include/Reconstruction/Mapper.h                   Mapper.cpp


)



add_library(
    reconstruction
    STATIC

    ../../include/Reconstruction/Utils.h                    Utils.cpp
    ../../include/Reconstruction/Projection.h               Projection.cpp


    ../../include/Reconstruction/Image.h                    Image.cpp
    ../../include/Reconstruction/Point2D.h                  Point2D.cpp
    ../../include/Reconstruction/Point3D.h                  Point3D.cpp
    ../../include/Reconstruction/Track.h                    Track.cpp


    ../../include/Reconstruction/SceneGraph.h               SceneGraph.cpp
    ../../include/Reconstruction/RegisterGraph.h            RegisterGraph.cpp


    ../../include/Reconstruction/Initializer.h              Initializer.cpp
    ../../include/Reconstruction/Registrant.h               Registrant.cpp
    ../../include/Reconstruction/Triangulator.h             Triangulator.cpp
    ../../include/Reconstruction/Map.h                      Map.cpp
    ../../include/Reconstruction/MapBuilder.h               MapBuilder.cpp


)

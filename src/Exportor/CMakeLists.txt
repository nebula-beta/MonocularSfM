
add_library(
    exportor
    STATIC
    ../../include/Exportor/Exportor.h                    Exportor.cpp

    ../../include/Exportor/OpenMVSInterface.h

    ../../include/Exportor/PLYExportor.h                 PLYExportor.cpp
    ../../include/Exportor/OpenMVSExportor.h             OpenMVSExportor.cpp
)


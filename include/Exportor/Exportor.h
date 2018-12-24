#ifndef __EXPORTOR_H__
#define __EXPORTOR_H__

#include <string>
#include "Reconstruction/Map.h"

namespace MonocularSfM
{
class Exportor
{
public:
    Exportor(const Map* const map);
    Export(const std::string& path, const std::string& filename);

private:
    std::string UnionPath(const std::string& path, const std::string& filename);
};


}// MonocularSfM

#endif //__EXPORTOR_H__

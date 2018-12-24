#ifndef __PLY_EXPORTOR_H__
#define __PLY_EXPORTOR_H__

#include <string>

namespace MonocularSfM
{
class PLYExportor
{
public:
    PLYExportor();
    virtual Export(const std::string& path, const std::string& filename) = 0;


};


}// MonocularSfM

#endif //__EXPORTOR_H__

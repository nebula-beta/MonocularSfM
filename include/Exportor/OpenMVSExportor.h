#ifndef __OPENMVS_EXPORTOR_H__
#define __OPENMVS_EXPORTOR_H__

#include <string>


namespace MonocularSfM
{
class OpenMVSExportor
{
public:
    OpenMVSExportor();
    Export(const std::string& path, const std::string& filename);

};


}// MonocularSfM

#endif //__OPENMVS_EXPORTOR_H__

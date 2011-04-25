#ifndef GMAC_API_OPENCL_LITE_MODEMAP_IMPL_H_
#define GMAC_API_OPENCL_LITE_MODEMAP_IMPL_H_

#include "api/opencl/lite/Mode.h"

namespace __impl { namespace opencl { namespace lite {

inline
ModeMap::ModeMap() :
    gmac::util::RWLock("ModeMap")
{ }

inline
ModeMap::~ModeMap()
{
    lockWrite();
    std::map<cl_context, Mode *>::const_iterator i;
    for(i = begin(); i != end(); i++) delete i->second;
    clear();
    unlock();
}

inline
bool ModeMap::insert(cl_context ctx, Mode &mode)
{
    lockWrite();
    std::pair<Parent::iterator, bool> ret = Parent::insert(Parent::value_type(ctx, &mode));
    unlock();
    return ret.second;
}

inline
void ModeMap::remove(cl_context ctx)
{
    lockWrite();
    Parent::erase(ctx);
    unlock();
}

inline
Mode *ModeMap::get(cl_context ctx) const
{
    Mode *ret = NULL;
    lockRead();
    Parent::const_iterator i = find(ctx);
    if(i != end()) {
        ret = i->second;
        ret->use();
    }
    unlock();
    return ret;
}

}}}
#endif

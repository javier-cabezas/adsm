#ifndef GMAC_UTIL_REFERENCE_IMPL_H_
#define GMAC_UTIL_REFERENCE_IMPL_H_

#include "Logger.h"

namespace __impl { namespace util {

inline Reference::Reference() :
    ref_(1)
{
}

inline Reference::~Reference()
{
}

inline gmacError_t Reference::cleanUp()
{
    return gmacSuccess;
}

inline void Reference::incRef() const
{
    AtomicInc(ref_);
}

inline void Reference::decRef()
{
    if (AtomicDec(ref_) == 0) {
        gmacError_t ret = cleanUp();
        ASSERTION(ret == gmacSuccess);
        delete this;
    }
}

}}
#endif

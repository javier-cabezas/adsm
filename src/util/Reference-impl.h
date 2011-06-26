#ifndef GMAC_UTIL_REFERENCE_IMPL_H_
#define GMAC_UTIL_REFERENCE_IMPL_H_

#include "Logger.h"

namespace __impl { namespace util {

inline
Reference::Reference(const char *name) :
    ref_(1)
#ifdef DEBUG
    , className_(name)
#endif
{
    TRACE(LOCAL, "Creating reference for %s: %p", className_.c_str(), this);
}

inline
Reference::~Reference()
{
    ASSERTION(ref_ == 0, "Expected 0, value = %d", ref_);
    TRACE(LOCAL, "Destroying reference for: %p", this);
}

inline gmacError_t Reference::cleanUp()
{
    return gmacSuccess;
}

inline void
Reference::incRef() const
{
#ifdef DEBUG
    Atomic a = AtomicInc(ref_);
    TRACE(LOCAL, "Incrementing reference for %s: %p = %d", className_.c_str(), this, a);
#else
    AtomicInc(ref_);
#endif
}

inline void
Reference::decRef()
{
    Atomic a = AtomicDec(ref_);
    ASSERTION(a >= 0);

    TRACE(LOCAL, "Decrementing reference for %s: %p = %d", className_.c_str(), this, a);

    if (a == 0) {
        TRACE(LOCAL, "0 references for %s: %p", className_.c_str(), this);
        gmacError_t ret = cleanUp();
        ASSERTION(ret == gmacSuccess);
        delete this;
    }
}

}}
#endif

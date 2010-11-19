#ifndef GMAC_UTIL_REFERENCE_IMPL_H_
#define GMAC_UTIL_REFERENCE_IMPL_H_

namespace __impl { namespace util {

inline Reference::Reference() : ref_(1) {};
inline Reference::~Reference() {};

inline void Reference::use() const
{
    AtomicInc(ref_);
}

inline void Reference::release() const
{
    if(AtomicDec(ref_) == 0) delete this;
}

}}
#endif

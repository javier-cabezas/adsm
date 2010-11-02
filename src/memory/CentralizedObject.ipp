#ifndef GMAC_MEMORY_CENTRALIZEDOBJECT_IPP_
#define GMAC_MEMORY_CENTRALIZEDOBJECT_IPP_

namespace gmac { namespace memory {

inline Mode &
CentralizedObject::owner() const
{
    return gmac::Mode::current();
}

inline bool
CentralizedObject::isLocal() const
{
    return false;
}

inline bool
CentralizedObject::isInAccelerator() const
{
    return false;
}

}}

#endif

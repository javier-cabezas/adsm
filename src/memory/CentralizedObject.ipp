#ifndef GMAC_MEMORY_CENTRALIZEDOBJECT_IPP_
#define GMAC_MEMORY_CENTRALIZEDOBJECT_IPP_

namespace __impl { namespace memory {

inline core::Mode &
CentralizedObject::owner() const
{
    return core::Mode::current();
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

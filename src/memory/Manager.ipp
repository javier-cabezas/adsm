#ifndef __GMAC_MEMORY_MANAGER_IPP
#define __GMAC_MEMORY_MANAGER_IPP

namespace gmac { namespace memory {

inline Protocol &
Manager::protocol() const
{
    return *_protocol;
}

}}

#endif

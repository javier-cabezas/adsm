#ifndef GMAC_MEMORY_MANAGER_IPP_
#define GMAC_MEMORY_MANAGER_IPP_

namespace __impl { namespace memory {

inline Protocol &
Manager::protocol() const
{
    return *protocol_;
}

}}

#endif

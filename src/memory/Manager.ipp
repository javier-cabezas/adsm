#ifndef GMAC_MEMORY_MANAGER_IPP_
#define GMAC_MEMORY_MANAGER_IPP_

namespace gmac { namespace memory { namespace __impl {

inline memory::Protocol &
Manager::protocol() const
{
    return *protocol_;
}

}}}

#endif

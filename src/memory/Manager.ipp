#ifndef __MANAGER_IPP_
#define __MANAGER_IPP_

namespace gmac { namespace memory {

inline Manager::~Manager()
{
    trace("Memory manager finishes");
    assertion(__count == 0);
    delete protocol;
}

#ifndef USE_MMAP
inline bool Manager::requireUpdate(Block *block)
{
    return protocol->requireUpdate(block);
}
#endif

}};
#endif

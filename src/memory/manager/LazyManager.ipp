#ifndef __MEMORY_LAZYMANAGER_IPP_
#define __MEMORY_LAZYMANAGER_IPP_

namespace gmac { namespace memory { namespace manager {

inline int
LazyManager::defaultProt()
{
    return PROT_READ;
}

}}}

#endif

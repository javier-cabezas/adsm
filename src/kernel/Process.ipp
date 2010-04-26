#ifndef __KERNEL_PROCESS_IPP_
#define __KERNEL_PROCESS_IPP_

#include "Process.h"

namespace gmac {


inline ContextList &
Process::contexts()
{
    return _contexts;
}

inline const void *
Process::translate(const void *addr)
{
    return (const void *)translate((void *)addr);
}

inline size_t
Process::totalMemory()
{
    return _totalMemory;
}

inline size_t
Process::accs() const
{
    return _accs.size();
}

inline
memory::RegionMap &Process::global()
{
    return __global;
}

inline
const memory::RegionMap &Process::global() const
{
    return __global;
}

inline
memory::RegionMap &Process::shared()
{
    return __shared;
}

inline
const memory::RegionMap &Process::shared() const
{
    return __shared;
}

}

#endif

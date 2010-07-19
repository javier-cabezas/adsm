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
Process::translate(const void *addr) const
{
    return (const void *)translate((void *)addr);
}

inline size_t
Process::totalMemory()
{
    return _totalMemory;
}

inline size_t
Process::nAccelerators() const
{
    return _accs.size();
}

inline
memory::RegionMap &Process::global()
{
    return _global;
}

inline
const memory::RegionMap &Process::global() const
{
    return _global;
}

inline
memory::RegionMap &Process::shared()
{
    return _shared;
}

inline
const memory::RegionMap &Process::shared() const
{
    return _shared;
}

}

#endif

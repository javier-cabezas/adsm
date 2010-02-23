#ifndef __KERNEL_PROCESS_IPP_
#define __KERNEL_PROCESS_IPP_

#include "Process.h"

namespace gmac {

#if 0
inline void *
SharedMemory::start() const
{
    return _addr;
}

inline size_t
SharedMemory::size() const
{
    return _size;
}

inline void
SharedMemory::inc()
{
    _count++;
}

inline size_t
SharedMemory::dec()
{
    return --_count;
}
#endif

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

}

#endif

#ifndef __KERNEL_PROCESS_IPP_
#define __KERNEL_PROCESS_IPP_

#include "Process.h"

namespace gmac {

inline ModeList &
Process::modes()
{
    return _modes;
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
memory::ObjectMap &Process::global()
{
    return __global;
}

inline
const memory::ObjectMap &Process::global() const
{
    return __global;
}

inline
memory::ObjectMap &Process::shared()
{
    return __shared;
}

inline
const memory::ObjectMap &Process::shared() const
{
    return __shared;
}

}

#endif

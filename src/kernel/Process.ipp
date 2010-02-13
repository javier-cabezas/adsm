#ifndef __KERNEL_PROCESS_IPP_
#define __KERNEL_PROCESS_IPP_

#include "Process.h"

namespace gmac {

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

inline const Process::ContextList &
Process::contexts() const
{
    return _contexts;
}

inline const void *
Process::translate(const void *addr)
{
    return (const void *)translate((void *)addr);
}

inline Process::SharedMap &
Process::sharedMem()
{
    return _sharedMem;
}

inline void
Process::addShared(void *addr, size_t size)
{
    std::pair<SharedMap::iterator, bool> ret =
        _sharedMem.insert(SharedMap::value_type(addr, SharedMemory(addr, size, _contexts.size())));
    if(ret.second == false) ret.first->second.inc();
}

inline bool
Process::removeShared(void *addr)
{
    SharedMap::iterator i;
    i = _sharedMem.find(addr);
    assert(i != _sharedMem.end());
    if(i->second.dec() == 0) {
        _sharedMem.erase(i);
        return true;
    }
    return false;
}

inline bool
Process::isShared(void *addr) const
{
    return _sharedMem.find(addr) != _sharedMem.end();
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

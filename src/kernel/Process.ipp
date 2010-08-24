#ifndef __KERNEL_PROCESS_IPP_
#define __KERNEL_PROCESS_IPP_

#include "Process.h"

namespace gmac {

inline void
QueueMap::cleanup()
{
    QueueMap::iterator q;
    lockWrite();
    for(q = Parent::begin(); q != Parent::end(); q++)
        delete q->second;
    clear();
    unlock();
}

inline std::pair<QueueMap::iterator, bool>
QueueMap::insert(THREAD_ID tid, ThreadQueue *q)
{
    lockWrite();
    std::pair<iterator, bool> ret =
        Parent::insert(value_type(tid, q));
    unlock();
    return ret;
}

inline QueueMap::iterator
QueueMap::find(THREAD_ID id)
{
    lockRead();
    iterator q = Parent::find(id);
    unlock();
    return q;
}

inline QueueMap::iterator
QueueMap::end()
{
    lockRead();
    iterator ret = Parent::end();
    unlock();
    return ret;
}
    

inline ModeList &
Process::modes()
{
    return __modes;
}

inline const void *
Process::translate(const void *addr) 
{
    return (const void *)translate((void *)addr);
}

inline size_t
Process::totalMemory()
{
    return __totalMemory;
}

inline size_t
Process::nAccelerators() const
{
    return __accs.size();
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

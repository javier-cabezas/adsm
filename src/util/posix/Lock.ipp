#ifndef __UTIL_POSIX_LOCK_IPP_
#define __UTIL_POSIX_LOCK_IPP_

#include <debug.h>
#include <threads.h>

#include <cassert>
#include <cstdio>

namespace gmac { namespace util {

inline
void ParaverLock::push()
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__pushState(
        paraver::trace->__pushEvent(*event, id), *exclusive);
#endif
}

inline
void ParaverLock::pop()
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__popState(
        paraver::trace->__pushEvent(*event, 0));
#endif
}

inline void
Lock::lock()
{
    push();
    pthread_mutex_lock(&__mutex);
    pop();
}

inline void
Lock::unlock()
{
    pthread_mutex_unlock(&__mutex);
}

inline void
RWLock::lockRead()
{
    push();
    pthread_rwlock_rdlock(&__lock);
    pop();
}

inline void
RWLock::lockWrite()
{
    push();
    pthread_rwlock_wrlock(&__lock);
    pop();
}

inline void
RWLock::unlock()
{
    pthread_rwlock_unlock(&__lock);
}

}}

#endif

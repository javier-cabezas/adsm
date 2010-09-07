#ifndef __UTIL_POSIX_LOCK_IPP_
#define __UTIL_POSIX_LOCK_IPP_

#include <debug.h>
#include <threads.h>

#include <cassert>
#include <cstdio>

namespace gmac { namespace util {

inline
void ParaverLock::enter()
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__pushState(
        paraver::trace->__pushEvent(*event, id), *exclusive);
#endif
}

inline
void ParaverLock::locked()
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__pushEvent(*event, 0);
#endif
}


inline
void ParaverLock::exit()
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__popState();
#endif
}


inline void
Lock::lock()
{
    enter();
    pthread_mutex_lock(&__mutex);
    locked();
}

inline void
Lock::unlock()
{
    exit();
    pthread_mutex_unlock(&__mutex);
}

inline void
RWLock::lockRead()
{
    exit();
    pthread_rwlock_rdlock(&__lock);
    locked();
}

inline void
RWLock::lockWrite()
{
    enter();
    pthread_rwlock_wrlock(&__lock);
    locked();
}

inline void
RWLock::unlock()
{
    exit();
    pthread_rwlock_unlock(&__lock);
}

}}

#endif

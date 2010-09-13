#ifndef __UTIL_POSIX_LOCK_IPP_
#define __UTIL_POSIX_LOCK_IPP_

#include <debug.h>
#include <threads.h>

#include <cassert>
#include <cstdio>

namespace gmac { namespace util {

inline
void ParaverLock::enter() const
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__pushState(
        paraver::trace->__pushEvent(*event, id), *exclusive);
#endif
}

inline
void ParaverLock::locked() const
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__pushEvent(*event, 0);
#endif
}


inline
void ParaverLock::exit() const
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__popState();
#endif
}


inline void
Lock::lock() const
{
    enter();
    pthread_mutex_lock(&__mutex);
    locked();
}

inline void
Lock::unlock() const
{
    exit();
    pthread_mutex_unlock(&__mutex);
}

inline void
RWLock::lockRead() const
{
    enter();
    pthread_rwlock_rdlock(&__lock);
    locked();
}

inline void
RWLock::lockWrite() const
{
    enter();
    pthread_rwlock_wrlock(&__lock);
    locked();
}

inline void
RWLock::unlock() const
{
    exit();
    pthread_rwlock_unlock(&__lock);
}

}}

#endif

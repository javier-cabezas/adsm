#ifndef GMAC_UTIL_POSIX_LOCK_IPP_
#define GMAC_UTIL_POSIX_LOCK_IPP_

#include "config/debug.h"
#include "config/threads.h"

#include <cassert>
#include <cstdio>

namespace gmac { namespace util {

inline
void ParaverLock::enter() const
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__pushEvent(*event, id);
#endif
}

inline
void ParaverLock::locked() const
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__pushState(
        paraver::trace->__pushEvent(*event, 0), *exclusive);
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
    pthread_mutex_lock(&mutex_);
    locked();
}

inline void
Lock::unlock() const
{
    exit();
    pthread_mutex_unlock(&mutex_);
}

inline void
RWLock::lockRead() const
{
    enter();
    pthread_rwlock_rdlock(&lock_);
    locked();
}

inline void
RWLock::lockWrite() const
{
    enter();
    pthread_rwlock_wrlock(&lock_);
    locked();
}

inline void
RWLock::unlock() const
{
    exit();
    pthread_rwlock_unlock(&lock_);
}

}}

#endif

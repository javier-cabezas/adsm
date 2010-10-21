#ifndef GMAC_UTIL_POSIX_LOCK_IPP_
#define GMAC_UTIL_POSIX_LOCK_IPP_

#include "config/debug.h"
#include "config/threads.h"

#include <cassert>
#include <cstdio>

namespace gmac { namespace util {

inline void
LockImpl::lock() const
{
    enter();
    pthread_mutex_lock(&mutex_);
    locked();
}

inline void
LockImpl::unlock() const
{
    exit();
    pthread_mutex_unlock(&mutex_);
}

inline void
RWLock::lockRead() const
{
    enter();
    pthread_rwlock_rdlock(&lock_);
    done();
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

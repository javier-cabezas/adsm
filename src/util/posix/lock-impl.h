#ifndef GMAC_UTIL_POSIX_LOCK_IMPL_H_
#define GMAC_UTIL_POSIX_LOCK_IMPL_H_

#include <cassert>
#include <cstdio>

namespace __impl { namespace util {

#if !defined(__APPLE__)
inline void
spinlock::lock() const
{
    enter();
    pthread_spin_lock(&spinlock_);
    locked();
}

inline void
spinlock::unlock() const
{
    exit();
    pthread_spin_unlock(&spinlock_);
}
#endif

inline void
mutex::lock() const
{
    enter();
    pthread_mutex_lock(&mutex_);
    locked();
}

inline void
mutex::unlock() const
{
    exit();
    pthread_mutex_unlock(&mutex_);
}

inline void
lock_rw::lockRead() const
{
    enter();
    pthread_rwlock_rdlock(&lock_);
    done();
}

inline void
lock_rw::lockWrite() const
{
    enter();
    pthread_rwlock_wrlock(&lock_);
    locked();
}

inline void
lock_rw::unlock() const
{
    exit();
    pthread_rwlock_unlock(&lock_);
}

}}

#endif

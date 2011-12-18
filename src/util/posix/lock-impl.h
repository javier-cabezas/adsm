#ifndef GMAC_UTIL_POSIX_LOCK_IMPL_H_
#define GMAC_UTIL_POSIX_LOCK_IMPL_H_

#include <cassert>
#include <cstdio>

namespace __impl { namespace util {

#if !defined(__APPLE__)
template <typename T>
inline
spinlock<T>::spinlock(const char *name) :
    __impl::util::lock__(name)
{
    pthread_spin_init(&spinlock_, PTHREAD_PROCESS_PRIVATE);
}

template <typename T>
inline
spinlock<T>::~spinlock()
{
    pthread_spin_destroy(&spinlock_);
}

template <typename T>
inline void
spinlock<T>::lock() const
{
    enter();
    pthread_spin_lock(&spinlock_);
    locked();
}

template <typename T>
inline void
spinlock<T>::unlock() const
{
    exit();
    pthread_spin_unlock(&spinlock_);
}
#endif

template <typename T>
inline
mutex<T>::mutex(const char *name) :
    __impl::util::lock__(name)
{
    pthread_mutex_init(&mutex_, NULL);
}

template <typename T>
inline
mutex<T>::~mutex()
{
    pthread_mutex_destroy(&mutex_);
}

template <typename T>
inline void
mutex<T>::lock() const
{
    enter();
    pthread_mutex_lock(&mutex_);
    locked();
}

template <typename T>
inline void
mutex<T>::unlock() const
{
    exit();
    pthread_mutex_unlock(&mutex_);
}

template <typename T>
inline void
lock_rw<T>::lock_read() const
{
    enter();
    pthread_rwlock_rdlock(&lock_);
    done();
}

template <typename T>
inline void
lock_rw<T>::lock_write() const
{
    enter();
    pthread_rwlock_wrlock(&lock_);
    locked();
}

template <typename T>
inline void
lock_rw<T>::unlock() const
{
    exit();
    pthread_rwlock_unlock(&lock_);
}

template <typename T>
inline
lock_rw<T>::lock_rw(const char *name) :
    __impl::util::lock__(name)
{
    pthread_rwlock_init(&lock_, NULL);
}

template <typename T>
inline
lock_rw<T>::~lock_rw()
{
    pthread_rwlock_destroy(&lock_);
}

}}

#endif

#ifndef GMAC_UTIL_LOCK_IMPL_H_
#define GMAC_UTIL_LOCK_IMPL_H_

#include "trace/Tracer.h"

namespace __impl { namespace util {

inline
void __Lock::enter() const
{
#if defined(USE_TRACE_LOCKS)
    trace::RequestLock(name_.c_str());
#endif
}

inline
void __Lock::locked() const
{
#if defined(USE_TRACE_LOCKS)
    trace::AcquireLockExclusive(name_.c_str());
    exclusive_ = true;
#endif
}

inline
void __Lock::done() const
{
#if defined(USE_TRACE_LOCKS)
    trace::AcquireLockShared(name_.c_str());
#endif
}


inline
void __Lock::exit() const
{
#if defined(USE_TRACE_LOCKS)
    trace::ExitLock(name_.c_str());
    exclusive_ = false;
#endif
}

template <typename T>
inline
scoped_lock<T>::scoped_lock(T &obj) :
    obj_(obj),
    owned_(true)
{
    obj_.lock();
}

template <typename T>
inline
scoped_lock<T>::scoped_lock(scoped_lock<T> &lock) :
    obj_(lock.obj_),
    owned_(lock.owned_)
{
    if (lock.owned_) lock.owned_ = false;
}

template <typename T>
inline
scoped_lock<T>::~scoped_lock()
{
    if (owned_ == true) {
        obj_.unlock();
    }
}

template <typename T>
inline T &
scoped_lock<T>::operator()()
{
    return *this;
}

template <typename T>
const T &
scoped_lock<T>::operator()() const
{
    return *this;
}

template <typename T>
inline
scoped_lock<T> &
scoped_lock<T>::operator=(scoped_lock<T> &lock)
{
    obj_ = lock.obj_;
    owned_ = lock.owned_;
    if (lock.owned_) lock.owned_ = false;

    return *this;
}

}}
#endif

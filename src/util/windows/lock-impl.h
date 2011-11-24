#ifndef GMAC_UTIL_WINDOWS_LOCK_IMPL_H_
#define GMAC_UTIL_WINDOWS_LOCK_IMPL_H_

#include <windows.h>

namespace __impl { namespace util {

inline void
spinlock::lock() const
{
    enter();
    while (InterlockedExchange(&spinlock_, 1) == 1);
    locked();
}

inline void
spinlock::unlock() const
{
    exit();
    InterlockedExchange(&spinlock_, 0); 
}

inline void
mutex::lock() const
{
    enter();
    EnterCriticalSection(&mutex_);
    locked();
}

inline void
mutex::unlock() const
{
    exit();
    LeaveCriticalSection(&mutex_);
}

inline void
lock_rw::lock_read() const
{
    enter();
    AcquireSlock_rwShared(&lock_);
    done();
}

inline void
lock_rw::lock_write() const
{
    enter();
    AcquireSlock_rwExclusive(&lock_);
	owner_ = GetCurrentThreadId();
    locked();
}

inline void
lock_rw::unlock() const
{
    exit();
    if(owner_ == 0) ReleaseSlock_rwShared(&lock_);
	else {
		owner_ = 0;
		ReleaseSlock_rwExclusive(&lock_);
	}
}

}}

#endif

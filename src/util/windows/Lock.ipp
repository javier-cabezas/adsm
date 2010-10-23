#ifndef __UTIL_WINDOWS_LOCK_IPP_
#define __UTIL_WINDOWS_LOCK_IPP_


namespace gmac { namespace util {

inline void
LockImpl::lock() const
{
    enter();
    EnterCriticalSection(&mutex_);
    locked();
}

inline void
LockImpl::unlock() const
{
    exit();
    LeaveCriticalSection(&mutex_);
}

inline void
RWLockImpl::lockRead() const
{
    enter();
    AcquireSRWLockShared(&lock_);
    done();
}

inline void
RWLockImpl::lockWrite() const
{
    enter();
    AcquireSRWLockExclusive(&lock_);
    locked();
}

inline void
RWLockImpl::unlock() const
{
    exit();
    if(owner_ == 0) ReleaseSRWLockShared(&lock_);
	else {
		owner_ = 0;
		ReleaseSRWLockExclusive(&lock_);
	}
}

}}

#endif

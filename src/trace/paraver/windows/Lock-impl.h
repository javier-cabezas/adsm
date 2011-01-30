#ifndef GMAC_TRACE_PARAVER_WINDOWS_LOCK_IMPL_H_
#define GMAC_TRACE_PARAVER_WINDOWS_LOCK_IMPL_H_

#include <windows.h>

namespace __impl { namespace trace { namespace paraver {

inline
Lock::Lock() :
{
    InitializeCriticalSection(&mutex_);
}

inline
Lock::~Lock()
{
    DeleteCriticalSection(&mutex_);
}

inline void
Lock::lockRead() const
{
    enter();
    AcquireSRWLockShared(&lock_);
    done();
}

inline void
Lock::lockWrite() const
{
    enter();
    AcquireSRWLockExclusive(&lock_);
	owner_ = GetCurrentThreadId();
    locked();
}

inline void
Lock::unlock() const
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

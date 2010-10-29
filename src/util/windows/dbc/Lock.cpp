#include "Lock.h"

namespace gmac { namespace util { namespace __dbc {

Lock::Lock(const char *name) :
    __impl::Lock(name),
    locked_(false),
    owner_(0)
{
	InitializeCriticalSection(&internal_);
}

Lock::~Lock()
{
    DeleteCriticalSection(&internal_);
}

void Lock::lock() const
{
    __impl::Lock::lock();

    EnterCriticalSection(&internal_);
    ENSURES(locked_ == false);
    locked_ = true;
    owner_ = GetCurrentThreadId();
    LeaveCriticalSection(&internal_);
}

void Lock::unlock() const
{
    EnterCriticalSection(&internal_);
    REQUIRES(locked_ == true);
    EXPECTS(owner_ == GetCurrentThreadId());
    owner_ = 0;
    locked_ = false;

    __impl::Lock::unlock();

    LeaveCriticalSection(&internal_);
}

RWLock::RWLock(const char *name) :
    __impl::RWLock(name),
    state_(Idle),
    writer_(0)
{
    InitializeCriticalSection(&internal_);
}

RWLock::~RWLock()
{
    DeleteCriticalSection(&internal_);
}

void RWLock::lockRead() const
{
    EnterCriticalSection(&internal_);
    REQUIRES(readers_.find(GetCurrentThreadId()) == readers_.end());
    LeaveCriticalSection(&internal_);

    __impl::RWLock::lockRead();

    EnterCriticalSection(&internal_);
    ENSURES(state_ == Idle || state_ == Read);
    state_ = Read;
    readers_.insert(GetCurrentThreadId());
    LeaveCriticalSection(&internal_);
}

void RWLock::lockWrite() const
{
    EnterCriticalSection(&internal_);
    REQUIRES(readers_.find(GetCurrentThreadId()) == readers_.end());
	REQUIRES(writer_ == 0);
    LeaveCriticalSection(&internal_);

    __impl::RWLock::lockWrite();

    EnterCriticalSection(&internal_);
    ENSURES(readers_.empty() == true);
    ENSURES(state_ == Idle);
    state_ = Write;
    writer_ = GetCurrentThreadId();
    LeaveCriticalSection(&internal_);
}

void RWLock::unlock() const
{
    EnterCriticalSection(&internal_);
    if(writer_ == GetCurrentThreadId()) {
        REQUIRES(readers_.empty() == true);
        REQUIRES(state_ == Write);
        state_ = Idle;
        writer_ = 0;
    }
    else {
        REQUIRES(readers_.erase(GetCurrentThreadId()) == 1);
        REQUIRES(state_ == Read);
        if(readers_.empty() == true) state_ = Idle;
    }

    __impl::RWLock::unlock();

    LeaveCriticalSection(&internal_);
}

}}}

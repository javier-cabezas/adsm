#include "Lock.h"

namespace gmac { namespace util {

LockTest::LockTest(const char *name) :
    LockImpl(name),
    locked_(false),
    owner_(0)
{
	InitializeCriticalSection(&internal_);
}

LockTest::~LockTest()
{
    DeleteCriticalSection(&internal_);
}

void LockTest::lock() const
{
    LockImpl::lock();

    EnterCriticalSection(&internal_);
    ENSURES(locked_ == false);
    locked_ = true;
    owner_ = GetCurrentThreadId();
    LeaveCriticalSection(&internal_);
}

void LockTest::unlock() const
{
    EnterCriticalSection(&internal_);
    REQUIRES(locked_ == true);
    EXPECTS(owner_ == GetCurrentThreadId());
    owner_ = 0;
    locked_ = false;

    LockImpl::unlock();

    LeaveCriticalSection(&internal_);
}

RWLockTest::RWLockTest(const char *name) :
    RWLockImpl(name),
    state_(Idle),
    writer_(0)
{
    InitializeCriticalSection(&internal_);
}

RWLockTest::~RWLockTest()
{
    DeleteCriticalSection(&internal_);
}

void RWLockTest::lockRead() const
{
    EnterCriticalSection(&internal_);
    REQUIRES(readers_.find(GetCurrentThreadId()) == readers_.end());
    LeaveCriticalSection(&internal_);

    RWLockImpl::lockRead();

    EnterCriticalSection(&internal_);
    ENSURES(state_ == Idle || state_ == Read);
    state_ = Read;
    readers_.insert(GetCurrentThreadId());
    LeaveCriticalSection(&internal_);
}

void RWLockTest::lockWrite() const
{
    EnterCriticalSection(&internal_);
    REQUIRES(readers_.find(GetCurrentThreadId()) == readers_.end());
	REQUIRES(writer_ == 0);
    LeaveCriticalSection(&internal_);

    RWLockImpl::lockWrite();

    EnterCriticalSection(&internal_);
    ENSURES(readers_.empty() == true);
    ENSURES(state_ == Idle);
    state_ = Write;
    writer_ = GetCurrentThreadId();
    LeaveCriticalSection(&internal_);
}

void RWLockTest::unlock() const
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

    RWLockImpl::unlock();

    LeaveCriticalSection(&internal_);
}

} }

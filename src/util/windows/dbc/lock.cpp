#ifdef USE_DBC

#include "mutex.h"

namespace __dbc { namespace util {

spinlock::spinlock(const char *name) :
    __impl::util::spinlock(name),
    locked_(false),
    owner_(0)
{
	InitializeCriticalSection(&internal_);
}

spinlock::~spinlock()
{
    DeleteCriticalSection(&internal_);
}

void spinlock::lock() const
{
    EnterCriticalSection(&internal_);
    REQUIRES(owner_ != GetCurrentThreadId());
    LeaveCriticalSection(&internal_);

    __impl::util::spinlock::lock();

    EnterCriticalSection(&internal_);
    ENSURES(locked_ == false);
    locked_ = true;
    owner_ = GetCurrentThreadId();
    LeaveCriticalSection(&internal_);
}

void spinlock::unlock() const
{
    EnterCriticalSection(&internal_);
    REQUIRES(locked_ == true);
    EXPECTS(owner_ == GetCurrentThreadId());
    owner_ = 0;
    locked_ = false;

    __impl::util::spinlock::unlock();

    LeaveCriticalSection(&internal_);
}


mutex::mutex(const char *name) :
    __impl::util::mutex(name),
    locked_(false),
    owner_(0)
{
	InitializeCriticalSection(&internal_);
}

mutex::~mutex()
{
    DeleteCriticalSection(&internal_);
}

void mutex::lock() const
{
    __impl::util::mutex::lock();

    EnterCriticalSection(&internal_);
    ENSURES(locked_ == false);
    locked_ = true;
    owner_ = GetCurrentThreadId();
    LeaveCriticalSection(&internal_);
}

void mutex::unlock() const
{
    EnterCriticalSection(&internal_);
    REQUIRES(locked_ == true);
    EXPECTS(owner_ == GetCurrentThreadId());
    owner_ = 0;
    locked_ = false;

    __impl::util::mutex::unlock();

    LeaveCriticalSection(&internal_);
}

lock_rw::lock_rw(const char *name) :
    __impl::util::lock_rw(name),
    state_(Idle),
    writer_(0)
{
    InitializeCriticalSection(&internal_);
}

lock_rw::~lock_rw()
{
    DeleteCriticalSection(&internal_);
}

void lock_rw::lockRead() const
{
    EnterCriticalSection(&internal_);
    REQUIRES(readers_.find(GetCurrentThreadId()) == readers_.end());
    LeaveCriticalSection(&internal_);

    __impl::util::lock_rw::lockRead();

    EnterCriticalSection(&internal_);
    ENSURES(state_ == Idle || state_ == Read);
    state_ = Read;
    readers_.insert(GetCurrentThreadId());
    LeaveCriticalSection(&internal_);
}

void lock_rw::lockWrite() const
{
    EnterCriticalSection(&internal_);
    REQUIRES(readers_.find(GetCurrentThreadId()) == readers_.end());
	REQUIRES(writer_ != GetCurrentThreadId());
    LeaveCriticalSection(&internal_);

    __impl::util::lock_rw::lockWrite();

    EnterCriticalSection(&internal_);
    ENSURES(readers_.empty() == true);
    ENSURES(state_ == Idle);
    state_ = Write;
    writer_ = GetCurrentThreadId();
    LeaveCriticalSection(&internal_);
}

void lock_rw::unlock() const
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

    __impl::util::lock_rw::unlock();

    LeaveCriticalSection(&internal_);
}

}}

#endif

#ifndef GMAC_UTIL_LOCK_IMPL_H_
#define GMAC_UTIL_LOCK_IMPL_H_

#include "trace/Tracer.h"

namespace __impl { namespace util {

inline
void __Lock::enter() const
{
#if defined(USE_TRACE)
    trace::SetThreadState(trace::Locked);
    trace::RequestLock(name_.c_str());
#endif
}

inline
void __Lock::locked() const
{
#if defined(USE_TRACE)
    trace::SetThreadState(trace::Exclusive);
    trace::AcquireLockExclusive(name_.c_str());
    exclusive_ = true;
#endif
}

inline
void __Lock::done() const
{
#if defined(USE_TRACE)
    trace::SetThreadState(trace::Running);
    trace::AcquireLockShared(name_.c_str());
#endif
}


inline
void __Lock::exit() const
{
#if defined(USE_TRACE)
    if(exclusive_) trace::SetThreadState(trace::Running);
    trace::ExitLock(name_.c_str());
    exclusive_ = false;
#endif
}

}}
#endif

#ifndef GMAC_UTIL_LOCK_IMPL_H_
#define GMAC_UTIL_LOCK_IMPL_H_

#include "trace/Tracer.h"

namespace gmac { namespace util {

inline
void __Lock::enter() const
{
#if defined(USE_TRACE)
	trace::SetThreadState(trace::Locked);
#endif
}

inline
void __Lock::locked() const
{
#if defined(USE_TRACE)
	trace::SetThreadState(trace::Exclusive);
#endif
}

inline
void __Lock::done() const
{
#if defined(USE_TRACE)
	trace::SetThreadState(trace::Running);
    exclusive_ = false;
#endif
}


inline
void __Lock::exit() const
{
#if defined(USE_TRACE)
	if(exclusive_) trace::SetThreadState(trace::Running);
#endif
}

}}
#endif

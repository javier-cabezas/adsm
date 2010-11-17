#ifndef __UTIL_POSIX_THREAD_IMPL_H_
#define __UTIL_POSIX_THREAD_IMPL_H_

namespace gmac { namespace util { 

inline THREAD_T GetThreadId()
{
	return pthread_self();
}

inline PROCESS_T GetProcessId()
{
	return getpid();
}

}}

#endif

#ifndef GMAC_UTIL_POSIX_THREAD_IMPL_H_
#define GMAC_UTIL_POSIX_THREAD_IMPL_H_

#include <sys/types.h>
#include <unistd.h>

namespace __impl { namespace util { 

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

#ifndef GMAC_UTIL_WINDOWS_THREAD_IMPL_H_
#define GMAC_UTIL_WINDOWS_THREAD_IMPL_H_

namespace __impl { namespace util { 

inline THREAD_T GetThreadId()
{
	return GetCurrentThreadId();
}

inline PROCESS_T GetProcessId()
{
	return GetCurrentProcessId();
}

}}

#endif

#ifndef __UTIL_WINDOWS_THREAD_IMPL_H_
#define __UTIL_WINDOWS_THREAD_IMPL_H_

namespace gmac { namespace util { 

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

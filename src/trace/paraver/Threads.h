#ifndef __PARAVER_THREADS_H_
#define __PARAVER_THREADS_H_

#if defined(POSIX)
#	include "posix/Threads.h"
#elif defined(WINDOWS)
#	include "windows/Threads.h"
#endif

#endif

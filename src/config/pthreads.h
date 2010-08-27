#ifndef __CONFIG_PTHREADS_H_
#define __CONFIG_PTHREADS_H_

#include <pthread.h>

#define THREAD_ID pthread_t
#define SELF() pthread_self()

#if defined(__LP64__) 
#define FMT_TID "0x%lx"
#else
#if defined(DARWIN)
#define FMT_TID "%p"
#else
#define FMT_TID "0x%lx"
#endif
#endif


#endif

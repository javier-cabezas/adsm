#ifndef GMAC_UTILS_H_
#define GMAC_UTILS_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#   include <windows.h>
    typedef DWORD thread_t;
#else
#   include <pthread.h>
    typedef pthread_t thread_t;
#endif

typedef void*(*thread_routine)(void *);
thread_t thread_create(thread_routine rtn, void *arg);
void thread_wait(thread_t id);

#ifdef __cplusplus
}
#endif

#endif

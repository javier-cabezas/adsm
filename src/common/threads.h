#ifndef __THREADS_H_
#define __THREADS_H_

#ifdef linux
#include <unistd.h>
#include <sys/syscall.h>

#define gettid() syscall(SYS_gettid)
#else

#define gettid() 0

#endif


#ifdef HAVE_LIBPTHREAD

#include <pthread.h>

#define MUTEX(mutex) pthread_mutex_t mutex 
#define MUTEX_INIT(mutex) pthread_mutex_init(&mutex, NULL)
#define MUTEX_DESTROY(mutex) pthread_mutex_destroy(&mutex)
#define MUTEX_LOCK(mutex) pthread_mutex_lock(&mutex)
#define MUTEX_TRYLOCK(mutex) pthread_mutex_try_lock(&mutex)
#define MUTEX_UNLOCK(mutex) pthread_mutex_unlock(&mutex)

#else
#warning "Thread-safe support not implemented"

#define MUTEX_INIT(mutex)
#define MUTEX_DESTROY(mutex)
#define MUTEX_LOCK(mutex)
#define MUTEX_TRYLOCK(mutex)
#define MUTEX_UNLOCK(mutex)

#endif

#endif

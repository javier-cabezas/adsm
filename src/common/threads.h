#ifndef __THREADS_H_
#define __THREADS_H_

#include <common/paraver.h>

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
#define __MUTEX_LOCK(mutex) pthread_mutex_lock(&mutex);
#define MUTEX_LOCK(mutex) \
	do {\
		pushState(_Waiting_);	\
		pthread_mutex_lock(&mutex);	\
		popState();	\
	} while(0)
#define MUTEX_TRYLOCK(mutex) pthread_mutex_try_lock(&mutex)
#define __MUTEX_UNLOCK(mutex) pthread_mutex_unlock(&mutex)
#define MUTEX_UNLOCK(mutex) \
	do {\
		pthread_mutex_unlock(&mutex);	\
	} while(0)
#else
#warning "Thread-safe support not implemented"

#define MUTEX_INIT(mutex)
#define MUTEX_DESTROY(mutex)
#define MUTEX_LOCK(mutex)
#define MUTEX_TRYLOCK(mutex)
#define MUTEX_UNLOCK(mutex)

#endif

#ifdef unix

#include <semaphore.h>

#define SEM(sem) sem_t sem;
#define SEM_INIT(sem, n) sem_init(&sem, 0, n)
#define SEM_DESTROY(sem) sem_destroy(&sem)
#define SEM_WAIT(sem) sem_wait(&sem)
#define SEM_POST(sem) sem_post(&sem)

#else
#warning "Thread-safe support not implemented"

#define SEM(sem) 
#define SEM_INIT(sem, n) 
#define SEM_DESTROY(sem)
#define SEM_WAIT(sem)
#define SEM_POST(sem)

#endif

#endif

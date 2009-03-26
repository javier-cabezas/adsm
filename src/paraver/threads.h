#ifndef __PARAVER_THREADS_H_
#define __PARAVER_THREADS_H_

#ifdef linux
#include <unistd.h>
#include <sys/syscall.h>

#define paraver_gettid() syscall(SYS_gettid)

#endif


#ifdef HAVE_LIBPTHREAD

#include <pthread.h>

#define PARAVER_MUTEX(mutex) pthread_mutex_t mutex 
#define PARAVER_MUTEX_INIT(mutex) pthread_mutex_init(&mutex, NULL)
#define PARAVER_MUTEX_DESTROY(mutex) pthread_mutex_destroy(&mutex)
#define PARAVER_MUTEX_LOCK(mutex) pthread_mutex_lock(&mutex)
#define PARAVER_MUTEX_TRYLOCK(mutex) pthread_mutex_try_lock(&mutex)
#define PARAVER_MUTEX_UNLOCK(mutex) pthread_mutex_unlock(&mutex)

#else
#warning "Thread-safe support not implemented"

#define PARAVER_MUTEX_INIT(mutex)
#define PARAVER_MUTEX_DESTROY(mutex)
#define PARAVER_MUTEX_LOCK(mutex)
#define PARAVER_MUTEX_TRYLOCK(mutex)
#define PARAVER_MUTEX_UNLOCK(mutex)

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

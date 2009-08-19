#ifndef __CONFIG_PTHREADS_H_
#define __CONFIG_PTHREADS_H_

#include <pthread.h>

#define MUTEX(mutex) pthread_mutex_t mutex 
#define MUTEX_INIT(mutex) pthread_mutex_init(&mutex, NULL)
#define MUTEX_DESTROY(mutex) pthread_mutex_destroy(&mutex)
#define MUTEX_LOCK(mutex) pthread_mutex_lock(&mutex);
#define MUTEX_TRYLOCK(mutex) pthread_mutex_try_lock(&mutex)
#define MUTEX_UNLOCK(mutex) pthread_mutex_unlock(&mutex)

#define PRIVATE(name) pthread_key_t name
#define PRIVATE_INIT(name, dtor) pthread_key_create(&name, dtor)
#define PRIVATE_SET(name, var) pthread_setspecific(name, var)
#define PRIVATE_GET(name) pthread_getspecific(name)

#endif

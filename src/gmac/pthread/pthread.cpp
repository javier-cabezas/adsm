#include <api/api.h>
#include <os/loader.h>

#include <paraver.h>
#include <debug.h>

#include <pthread.h>


SYM(int, __pthread_create, pthread_t *__restrict, __const pthread_attr_t *, void *(*)(void *), void *);

static void __attribute__((constructor(101))) gmacPthreadInit(void)
{
	LOAD_SYM(__pthread_create, pthread_create);
}

typedef struct {
	void *(*__start_routine)(void *);
	void *__arg;
} gmac_thread_t;

static gmac_thread_t gthread;
static void *gmac_pthread(void *arg) 
{
	gmac_thread_t *gthread = (gmac_thread_t *)arg;
	addThread();
	pushState(Running);
	gmacCreateManager();
	void *ret = gthread->__start_routine(gthread->__arg);
	gmacRemoveManager();
	popState();
	return ret;
}

int pthread_create(pthread_t *__restrict __newthread,
		__const pthread_attr_t *__restrict __attr, 
		void *(*__start_routine)(void *),
		void *__restrict __arg) 
{
	int ret = 0;
	pushState(ThreadCreate);
	TRACE("pthread_create");
	gthread.__start_routine = __start_routine;
	gthread.__arg = __arg;
	ret = __pthread_create(__newthread, __attr, gmac_pthread, (void *)&gthread);
	popState();
	return ret;
}

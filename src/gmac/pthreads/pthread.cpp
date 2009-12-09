#include <init.h>
#include <os/loader.h>
#include <kernel/Process.h>
#include <kernel/Context.h>

#include <order.h>
#include <paraver.h>
#include <debug.h>

#include <pthread.h>


SYM(int, __pthread_create, pthread_t *__restrict, __const pthread_attr_t *, void *(*)(void *), void *);

static pthread_mutex_t create_mutex = PTHREAD_MUTEX_INITIALIZER;

static void __attribute__((constructor(INTERPOSE))) gmacPthreadInit(void)
{
	LOAD_SYM(__pthread_create, pthread_create);
}

typedef struct {
	gmac::Context *__current;
	void *(*__start_routine)(void *);
	void *__arg;
} gmac_thread_t;


//static gmac_thread_t gthread;
static void *gmac_pthread(void *arg) 
{
	__enterGmac();
	gmac_thread_t *gthread = (gmac_thread_t *)arg;
	addThread();
	pushState(Init);
	proc->clone(gthread->__current);
	pthread_mutex_unlock(&create_mutex);
	popState();
	pushState(Running);
	__exitGmac();
	void *ret = gthread->__start_routine(gthread->__arg);
	__enterGmac();
	popState();
	// IG: commented out because it was producing segmentation
	// faults. I have no idea why this faults are being triggered
	//gmac::Context::current()->destroy();
	free(gthread);
	__exitGmac();
	return ret;
}

int pthread_create(pthread_t *__restrict __newthread,
		__const pthread_attr_t *__restrict __attr, 
		void *(*__start_routine)(void *),
		void *__restrict __arg) 
{
	int ret = 0;
	__enterGmac();
	pushState(ThreadCreate);
	TRACE("pthread_create");
	gmac_thread_t *gthread = (gmac_thread_t *)malloc(sizeof(gmac_thread_t));
	gthread->__current = gmac::Context::current();
	gthread->__start_routine = __start_routine;
	gthread->__arg = __arg;
	pthread_mutex_lock(&create_mutex);
	ret = __pthread_create(__newthread, __attr, gmac_pthread, (void *)gthread);
	pthread_mutex_lock(&create_mutex);
	pthread_mutex_unlock(&create_mutex);
	popState();
	__exitGmac();
	return ret;
}

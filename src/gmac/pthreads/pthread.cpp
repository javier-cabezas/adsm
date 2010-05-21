#include <init.h>
#include <os/loader.h>
#include <kernel/Process.h>
#include <kernel/Context.h>
#include <util/Lock.h>

#include <order.h>
#include <paraver.h>

#include <pthread.h>

static gmac::util::Lock *pLock;

SYM(int, __pthread_create, pthread_t *__restrict, __const pthread_attr_t *, void *(*)(void *), void *);

void threadInit(void)
{
	pLock = new gmac::util::Lock(LockPthread);
	LOAD_SYM(__pthread_create, pthread_create);
}

static void __attribute__((destructor())) gmacPthreadFini(void)
{
	delete pLock;
}

struct gmac_thread_t {
	void *(*__start_routine)(void *);
	void *__arg;
};

//static gmac_thread_t gthread;
static void *gmac_pthread(void *arg)
{
	__enterGmac();
	gmac_thread_t *gthread = (gmac_thread_t *)arg;
    proc->initThread();
	addThread();
    gmac::Context::initThread();
	pLock->unlock();
	pushState(Running);
	__exitGmac();
	void *ret = gthread->__start_routine(gthread->__arg);
	__enterGmac();
	popState();
    // Context already destroyed in Process destructor
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
    gmac::util::Logger::Trace("New POSIX thread");
	gmac_thread_t *gthread = (gmac_thread_t *)malloc(sizeof(gmac_thread_t));
	gthread->__start_routine = __start_routine;
	gthread->__arg = __arg;
	pLock->lock();
	ret = __pthread_create(__newthread, __attr, gmac_pthread, (void *)gthread);
	pLock->lock();
	pLock->unlock();
	popState();
	__exitGmac();
	return ret;
}

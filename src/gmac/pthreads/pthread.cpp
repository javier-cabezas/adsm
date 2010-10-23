#include "config/common.h"
#include "core/Process.h"
#include "core/Context.h"
#include "gmac/init.h"
#include "os/posix/loader.h"
#include "trace/Thread.h"
#include "util/Lock.h"

#include "config/order.h"

#include <pthread.h>

class GMAC_LOCAL ThreadLock : public gmac::util::Lock {
public:
    ThreadLock() : gmac::util::Lock("Thread") {};

    void lock() { gmac::util::Lock::lock(); }
    void unlock() { gmac::util::Lock::unlock(); }
};

static ThreadLock *pLock;

SYM(int, __pthread_create, pthread_t *__restrict, __const pthread_attr_t *, void *(*)(void *), void *);

void threadInit(void)
{
	pLock = new ThreadLock;
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
    gmac::trace::Thread::start();
	gmac::enterGmac();
	gmac_thread_t *gthread = (gmac_thread_t *)arg;
    gmac::Process &proc = gmac::Process::getInstance();
    proc.initThread();
	pLock->unlock();
    gmac::trace::Thread::run();
	gmac::exitGmac();
	void *ret = gthread->__start_routine(gthread->__arg);
	gmac::enterGmac();
    gmac::trace::Thread::resume();
    // Modes and Contexts already destroyed in Process destructor
    proc.finiThread();
	free(gthread);
	gmac::exitGmac();
	return ret;
}

int pthread_create(pthread_t *__restrict __newthread,
                   __const pthread_attr_t *__restrict __attr,
                   void *(*__start_routine)(void *),
                   void *__restrict __arg)
{
	int ret = 0;
	gmac::enterGmac();
    gmac::util::Logger::TRACE("New POSIX thread");
	gmac_thread_t *gthread = (gmac_thread_t *)malloc(sizeof(gmac_thread_t));
	gthread->__start_routine = __start_routine;
	gthread->__arg = __arg;
	pLock->lock();
	ret = __pthread_create(__newthread, __attr, gmac_pthread, (void *)gthread);
	pLock->lock();
	pLock->unlock();
	gmac::exitGmac();
	return ret;
}

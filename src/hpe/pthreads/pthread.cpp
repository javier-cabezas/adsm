/**
 * \file pthread.cpp
 *
 * Initialization routines for pthreads
 */

#include <pthread.h>

#include "config/common.h"
#include "config/order.h"

#include "core/hpe/Process.h"

#include "hpe/init.h"

#include "trace/Tracer.h"

#include "util/loader.h"
#include "util/Lock.h"


using namespace __impl::core::hpe;

SYM(int, __pthread_create, pthread_t *__restrict, __const pthread_attr_t *, void *(*)(void *), void *);

void threadInit(void)
{
    LOAD_SYM(__pthread_create, pthread_create);
}

static void __attribute__((destructor())) gmacPthreadFini(void)
{
}

struct gmac_thread_t {
    void *(*__start_routine)(void *);
    void *__arg;
    bool __do_exit;
};

static void *gmac_pthread(void *arg)
{
    gmac::trace::StartThread("CPU");
    enterGmac();
    gmac_thread_t *gthread = (gmac_thread_t *)arg;
    bool do_exit = gthread->__do_exit;
    Process &proc = getProcess();
    proc.initThread();
    gmac::trace::SetThreadState(gmac::trace::Running);
    if(do_exit) exitGmac();
    void *ret = gthread->__start_routine(gthread->__arg);
    if(do_exit) enterGmac();
    // Modes and Contexts already destroyed in Process destructor
    proc.finiThread();
    free(gthread);
    gmac::trace::SetThreadState(gmac::trace::Idle);
    exitGmac();
    return ret;
}

int pthread_create(pthread_t *__restrict __newthread,
                   __const pthread_attr_t *__restrict __attr,
                   void *(*__start_routine)(void *),
                   void *__restrict __arg)
{
    int ret = 0;
    bool externCall = inGmac() == 0;
    if(externCall) enterGmac();
    TRACE(GLOBAL, "New POSIX thread");
    gmac_thread_t *gthread = (gmac_thread_t *)malloc(sizeof(gmac_thread_t));
    gthread->__start_routine = __start_routine;
    gthread->__arg = __arg;
    gthread->__do_exit = externCall;
    ret = __pthread_create(__newthread, __attr, gmac_pthread, (void *)gthread);
    if(externCall) exitGmac();
    return ret;
}

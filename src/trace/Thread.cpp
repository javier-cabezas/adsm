#include "Thread.h"
#include <util/Logger.h>
#include <cassert>

namespace gmac { namespace trace {
#ifdef PARAVER
const char *Thread::initName = "Init";
paraver::StateName *Thread::initState = NULL;

const char *Thread::ioName = "IO";
paraver::StateName *Thread::ioState = NULL;

bool Thread::sanityChecks()
{
    if(paraver::trace == NULL) return false;
    if(initState == NULL) {
        initState = paraver::Factory<paraver::StateName>::create(initName);
    }
    if(ioState == NULL)
        ioState = paraver::Factory<paraver::StateName>::create(ioName);
    return true;
}

#endif

void Thread::start()
{
#ifdef PARAVER
    if(sanityChecks() == false) return;
    paraver::trace->__addThread();
    paraver::trace->__pushState(*paraver::Idle);
#endif
}

void Thread::start(THREAD_ID tid)
{
#ifdef PARAVER
    if(sanityChecks() == false) return;
    paraver::trace->__addThread(tid + offset);
    paraver::trace->__pushState(*paraver::Idle, tid + offset);
#endif
}

void Thread::end(THREAD_ID tid)
{
#ifdef PARAVER
    if(sanityChecks() == false) return;
    paraver::trace->__pushState(*paraver::Idle, tid + offset);
#endif
}

void Thread::run()
{
#ifdef PARAVER
    if(sanityChecks() == false) return;
    paraver::trace->__pushState(*paraver::Running);
#endif
}

void Thread::run(THREAD_ID tid)
{
#ifdef PARAVER
    if(sanityChecks() == false) return;
    paraver::trace->__pushState(*paraver::Running, tid + offset);
#endif
}

void Thread::init(THREAD_ID tid)
{
#ifdef PARAVER
    if(sanityChecks() == false) return;
    paraver::trace->__pushState(*initState, tid + offset);
#endif
}

void Thread::io()
{
#ifdef PARAVER
    if(sanityChecks() == false) return;
    paraver::trace->__pushState(*ioState);
#endif
}

void Thread::io(THREAD_ID tid)
{
#ifdef PARAVER
    if(sanityChecks() == false) return;
    paraver::trace->__pushState(*ioState, tid + offset);
#endif
}

void Thread::resume()
{
#ifdef PARAVER
    if(sanityChecks() == false) return;
    paraver::trace->__popState();
#endif
}

void Thread::resume(THREAD_ID tid)
{
#ifdef PARAVER
    if(sanityChecks() == false) return;
    paraver::trace->__popState((uint32_t)tid + offset);
#endif
}

}}

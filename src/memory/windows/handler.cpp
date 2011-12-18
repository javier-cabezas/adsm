#include <csignal>
#include <cerrno>

#include "config/common.h"
#include "core/process.h"
#include "memory/Handler.h"
#include "memory/Manager.h"
#include "trace/Tracer.h"

namespace __impl { namespace memory {

unsigned handler::Count_ = 0;

static core::process *Process_ = NULL;
static manager *Manager_ = NULL;

static LONG CALLBACK segvHandler(EXCEPTION_POINTERS *ex)
{
    /* Check that we are getting an access violation exception */
    if(ex->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION)
        return EXCEPTION_CONTINUE_SEARCH;

    if(Process_ == NULL || Manager_ == NULL) return EXCEPTION_CONTINUE_SEARCH;

    handler::Entry();
    trace::EnterCurrentFunction();

    bool writeAccess = false;
    if(ex->ExceptionRecord->ExceptionInformation[0] == 1) writeAccess = true;
    else if(ex->ExceptionRecord->ExceptionInformation[0] != 0) { handler::Exit(); return EXCEPTION_CONTINUE_SEARCH; }

    void *addr = (void *)ex->ExceptionRecord->ExceptionInformation[1];

    if(writeAccess == false) TRACE(GLOBAL, "Read SIGSEGV for %p", addr);
    else TRACE(GLOBAL, "Write SIGSEGV for %p", addr);

    bool resolved = false;
    core::Mode *mode = Process_->get_owner((const hostptr_t)addr);
    if(mode != NULL) {
        if(!writeAccess) resolved = Manager_->signal_read(*mode, (hostptr_t)addr);
        else             resolved = Manager_->signal_write(*mode, (hostptr_t)addr);
    }

    if(resolved == false) {
        fprintf(stderr, "Uoops! I could not find a mapping for %p. I will abort the execution\n", addr);
        abort();
        handler::Exit();
        return EXCEPTION_CONTINUE_SEARCH;
    }

    trace::ExitCurrentFunction();
    handler::Exit();

    return EXCEPTION_CONTINUE_EXECUTION;
}

void handler::setHandler()
{
    AddVectoredExceptionHandler(1, segvHandler);

    Handler_ = this;
    TRACE(GLOBAL, "New signal handler programmed");
}

void handler::restoreHandler()
{
        RemoveVectoredExceptionHandler(segvHandler);

        Handler_ = NULL;
        TRACE(GLOBAL, "Old signal handler restored");
}

void handler::setProcess(core::process &proc)
{
    Process_ = &proc;
}

void handler::setManager(manager &manager)
{
    Manager_ = &manager;
}



}}

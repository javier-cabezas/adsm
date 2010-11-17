#include <csignal>
#include <cerrno>

#include "gmac/init.h"
#include "memory/Handler.h"
#include "memory/Manager.h"
#include "trace/Tracer.h"

namespace gmac { namespace memory {



unsigned Handler::Count_ = 0;

static LONG CALLBACK segvHandler(EXCEPTION_POINTERS *ex)
{
	/* Check that we are getting an access violation exception */
	if(ex->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION)
		return EXCEPTION_CONTINUE_SEARCH;

	enterGmac();
	trace::EnterCurrentFunction();
	
	bool writeAccess = false;
	if(ex->ExceptionRecord->ExceptionInformation[0] == 1) writeAccess = true;
	else if(ex->ExceptionRecord->ExceptionInformation[0] != 0) { exitGmac(); return EXCEPTION_CONTINUE_SEARCH; }
	
	void *addr = (void *)ex->ExceptionRecord->ExceptionInformation[1];

	if(writeAccess == false) TRACE(GLOBAL, "Read SIGSEGV for %p", addr);
	else TRACE(GLOBAL, "Write SIGSEGV for %p", addr);

	bool resolved = false;
	Manager &manager = Manager::getInstance();
	if(!writeAccess) resolved = manager.read(addr);
	else resolved = manager.write(addr);

	if(resolved == false) {
		fprintf(stderr, "Uoops! I could not find a mapping for %p. I will abort the execution\n", addr);
		abort();
		exitGmac();
		return EXCEPTION_CONTINUE_SEARCH;
	}

	trace::ExitCurrentFunction();
	exitGmac();

	return EXCEPTION_CONTINUE_EXECUTION;
}

void Handler::setHandler() 
{
	AddVectoredExceptionHandler(1, segvHandler);

	Handler_ = this;
	TRACE(GLOBAL, "New signal handler programmed");
}

void Handler::restoreHandler()
{
	RemoveVectoredExceptionHandler(segvHandler);

	Handler_ = NULL;
	TRACE(GLOBAL, "Old signal handler restored");
}


}}

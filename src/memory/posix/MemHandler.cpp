#include <memory/MemHandler.h>
#include <memory/ProtRegion.h>

#include <debug.h>
#include <paraver.h>

namespace gmac {

struct sigaction MemHandler::defaultAction;
MemHandler *handler = NULL;
unsigned MemHandler::count = 0;

void MemHandler::setHandler() 
{
	struct sigaction segvAction;
	memset(&segvAction, 0, sizeof(segvAction));
	segvAction.sa_sigaction = segvHandler;
	segvAction.sa_flags = SA_SIGINFO | SA_RESTART;
	sigemptyset(&segvAction.sa_mask);

	if(sigaction(SIGSEGV, &segvAction, &defaultAction) < 0)
		FATAL("sigaction: %s", strerror(errno));

	handler = this;
	TRACE("New SIGSEGV handler programmed");
}

void MemHandler::restoreHandler()
{
	if(sigaction(SIGSEGV, &defaultAction, NULL) < 0)
		FATAL("sigaction: %s", strerror(errno));

	handler = NULL;
	TRACE("Old SIGSEGV handler restored");
}

void MemHandler::segvHandler(int s, siginfo_t *info, void *ctx)
{
	enterFunction(gmacSignal);
	mcontext_t *mCtx = &((ucontext_t *)ctx)->uc_mcontext;
	unsigned long writeAccess = mCtx->gregs[REG_ERR] & 0x2;

	if(!writeAccess) TRACE("Read SIGSEGV for %p", info->si_addr);
	else TRACE("Write SIGSEGV for %p", info->si_addr);

	bool resolved = false;
	if(!writeAccess) resolved = handler->read(info->si_addr);
	else resolved = handler->write(info->si_addr);

	if(resolved == false) {
		TRACE("SIGSEGV for NULL Region");
		abort();
		// TODO: set the signal mask and other stuff
		if(defaultAction.sa_flags & SA_SIGINFO)
			return defaultAction.sa_sigaction(s, info, ctx);
		return defaultAction.sa_handler(s);
	}

	TRACE("SIGSEGV done");
	exitFunction();
}

}

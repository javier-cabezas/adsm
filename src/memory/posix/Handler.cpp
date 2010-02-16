#include <gmac/init.h>

#include <memory/Handler.h>
#include <memory/ProtRegion.h>

#include <debug.h>
#include <paraver.h>

namespace gmac { namespace memory {

struct sigaction Handler::defaultAction;
Handler *handler = NULL;
unsigned Handler::count = 0;

void Handler::setHandler() 
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

void Handler::restoreHandler()
{
	if(sigaction(SIGSEGV, &defaultAction, NULL) < 0)
		FATAL("sigaction: %s", strerror(errno));

	handler = NULL;
	TRACE("Old SIGSEGV handler restored");
}

void Handler::segvHandler(int s, siginfo_t *info, void *ctx)
{
	__enterGmac();
	enterFunction(gmacSignal);
	mcontext_t *mCtx = &((ucontext_t *)ctx)->uc_mcontext;

#if defined(LINUX)
	unsigned long writeAccess = mCtx->gregs[REG_ERR] & 0x2;
#elif defined(DARWIN)
	unsigned long writeAccess = (*mCtx)->__es.__err & 0x2;
#endif

	if(!writeAccess) TRACE("Read SIGSEGV for %p", info->si_addr);
	else TRACE("Write SIGSEGV for %p", info->si_addr);

	bool resolved = false;
	if(!writeAccess) resolved = handler->read(info->si_addr);
	else resolved = handler->write(info->si_addr);

	if(resolved == false) {
		fprintf(stderr, "Uoops! I could not find a mapping for %p. I will abort the execution\n", info->si_addr);
		abort();
		// TODO: set the signal mask and other stuff
		if(defaultAction.sa_flags & SA_SIGINFO) 
			return defaultAction.sa_sigaction(s, info, ctx);
		return defaultAction.sa_handler(s);
	}

	TRACE("SIGSEGV done");
	exitFunction();
	__exitGmac();
}

} }

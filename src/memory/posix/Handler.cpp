#include <gmac/init.h>

#include <memory/Handler.h>
#include <memory/ProtRegion.h>

#include <paraver.h>

#include <cerrno>

namespace gmac { namespace memory {

struct sigaction Handler::defaultAction;
Handler *handler = NULL;
unsigned Handler::count = 0;

#if defined(LINUX)
int Handler::signum = SIGSEGV;
#elif defined(DARWIN)
int Handler::signum = SIGBUS;
#endif

void Handler::setHandler() 
{
	struct sigaction segvAction;
	memset(&segvAction, 0, sizeof(segvAction));
	segvAction.sa_sigaction = segvHandler;
	segvAction.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&segvAction.sa_mask);

	if(sigaction(signum, &segvAction, &defaultAction) < 0)
		util::Logger::fatal("sigaction: %s", strerror(errno));

	handler = this;
	util::Logger::Trace("New signal handler programmed");
}

void Handler::restoreHandler()
{
	if(sigaction(signum, &defaultAction, NULL) < 0)
		util::Logger::fatal("sigaction: %s", strerror(errno));

	handler = NULL;
	util::Logger::trace("Old signal handler restored");
}

void Handler::segvHandler(int s, siginfo_t *info, void *ctx)
{
	__enterGmac();
	enterFunction(FuncGmacSignal);
	mcontext_t *mCtx = &((ucontext_t *)ctx)->uc_mcontext;

#if defined(LINUX)
	unsigned long writeAccess = mCtx->gregs[REG_ERR] & 0x2;
#elif defined(DARWIN)
	unsigned long writeAccess = (*mCtx)->__es.__err & 0x2;
#endif
    void * addr = info->si_addr;

	if(!writeAccess) util::Logger::Trace("Read SIGSEGV for %p", addr);
	else util::Logger::Trace("Write SIGSEGV for %p", addr);

	bool resolved = false;
	if(!writeAccess) resolved = handler->read(addr);
	else resolved = handler->write(addr);

	if(resolved == false) {
		fprintf(stderr, "Uoops! I could not find a mapping for %p. I will abort the execution\n", addr);
		abort();
		// TODO: set the signal mask and other stuff
		if(defaultAction.sa_flags & SA_SIGINFO) 
			return defaultAction.sa_sigaction(s, info, ctx);
		return defaultAction.sa_handler(s);
	}

	util::Logger::Trace("SIGSEGV done");
	exitFunction();
	__exitGmac();
}

}}

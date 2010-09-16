#include <gmac/init.h>
#include <memory/Handler.h>
#include <memory/Manager.h>
#include <trace/Function.h>

#include <signal.h>
#include <cerrno>

namespace gmac { namespace memory {

struct sigaction defaultAction;
Handler *handler = NULL;
unsigned Handler::count = 0;

#if defined(LINUX)
int Handler::signum = SIGSEGV;
#elif defined(DARWIN)
int Handler::signum = SIGBUS;
#endif

static void segvHandler(int s, siginfo_t *info, void *ctx)
{
	__enterGmac();
	trace::Function::start("GMAC", "gmacSignal");
	mcontext_t *mCtx = &((ucontext_t *)ctx)->uc_mcontext;

#if defined(LINUX)
	unsigned long writeAccess = mCtx->gregs[REG_ERR] & 0x2;
#elif defined(DARWIN)
	unsigned long writeAccess = (*mCtx)->__es.__err & 0x2;
#endif
    void * addr = info->si_addr;

	if(!writeAccess) gmac::util::Logger::TRACE("Read SIGSEGV for %p", addr);
	else gmac::util::Logger::TRACE("Write SIGSEGV for %p", addr);

	bool resolved = false;
	if(!writeAccess) resolved = manager->read(addr);
	else resolved = manager->write(addr);

	if(resolved == false) {
		fprintf(stderr, "Uoops! I could not find a mapping for %p. I will abort the execution\n", addr);
		abort();
		// TODO: set the signal mask and other stuff
		if(defaultAction.sa_flags & SA_SIGINFO) 
			return defaultAction.sa_sigaction(s, info, ctx);
		return defaultAction.sa_handler(s);
	}

	gmac::util::Logger::TRACE("SIGSEGV done");
	trace::Function::end("GMAC");
	__exitGmac();
}


void Handler::setHandler() 
{
	struct sigaction segvAction;
	memset(&segvAction, 0, sizeof(segvAction));
	segvAction.sa_sigaction = segvHandler;
	segvAction.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&segvAction.sa_mask);

	if(sigaction(signum, &segvAction, &defaultAction) < 0)
		gmac::util::Logger::Fatal("sigaction: %s", strerror(errno));

	handler = this;
	gmac::util::Logger::TRACE("New signal handler programmed");
}

void Handler::restoreHandler()
{
	if(sigaction(signum, &defaultAction, NULL) < 0)
		gmac::util::Logger::Fatal("sigaction: %s", strerror(errno));

	handler = NULL;
	gmac::util::Logger::TRACE("Old signal handler restored");
}


}}

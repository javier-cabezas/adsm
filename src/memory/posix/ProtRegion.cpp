#include <ProtRegion.h>

#include <debug.h>
#include <paraver.h>

#include <string.h>

#include <algorithm>

namespace gmac {
struct sigaction ProtRegion::defaultAction;

void ProtRegion::setHandler()
{
	struct sigaction segvAction;
	memset(&segvAction, 0, sizeof(segvAction));
	segvAction.sa_sigaction = segvHandler;
	segvAction.sa_flags = SA_SIGINFO | SA_RESTART;
	sigemptyset(&segvAction.sa_mask);

	if(sigaction(SIGSEGV, &segvAction, &defaultAction) < 0)
		FATAL("sigaction: %s", strerror(errno));

	TRACE("New SIGSEGV handler programmed");
}

void ProtRegion::restoreHandler()
{
	if(sigaction(SIGSEGV, &defaultAction, NULL) < 0)
		FATAL("sigaction: %s", strerror(errno));

	TRACE("Old SIGSEGV handler restored");
}

void ProtRegion::segvHandler(int s, siginfo_t *info, void *ctx)
{
	enterFunction(_gmacSignal_);
	mcontext_t *mCtx = &((ucontext_t *)ctx)->uc_mcontext;
	unsigned long writeAccess = mCtx->gregs[REG_ERR] & 0x2;

	if(!writeAccess) TRACE("Read SIGSEGV for %p", info->si_addr);
	else TRACE("Write SIGSEGV for %p", info->si_addr);

	ProtRegion *r = MemHandler::get()->find(info->si_addr);
	if(r == NULL || r->isOwner() == false) {
		if(r == NULL) { TRACE("SIGSEGV for NULL Region"); }
		else { TRACE("SIGSEGV for external Region"); }
		abort();
		// TODO: set the signal mask and other stuff
		if(defaultAction.sa_flags & SA_SIGINFO)
			return defaultAction.sa_sigaction(s, info, ctx);
		return defaultAction.sa_handler(s);
	}
	TRACE("SIGSEGV for shared memory");

	if(!writeAccess) r->read(info->si_addr);
	else r->write(info->si_addr);

	TRACE("SIGSEGV done");
	exitFunction();
}

};

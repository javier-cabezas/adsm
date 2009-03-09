#include <common/MemRegion.h>
#include <common/debug.h>

#include <string.h>

#include <algorithm>

namespace gmac {
struct sigaction ProtRegion::defaultAction;

void ProtRegion::setHandler()
{
	struct sigaction segvAction;
	memset(&segvAction, 0, sizeof(segvAction));
	segvAction.sa_sigaction = segvHandler;
	segvAction.sa_flags = SA_SIGINFO;
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
	std::list<ProtRegion *>::iterator i;
	TRACE("SIGSEGV for %p", info->si_addr);
	i = std::find_if(regionList.begin(), regionList.end(),
		FindMem(info->si_addr));
	if(i == regionList.end()) {
		// The signal was not caused by us
		// TODO: set the signal mask and other stuff
		if(defaultAction.sa_flags & SA_SIGINFO)
			return defaultAction.sa_sigaction(s, info, ctx);
		return defaultAction.sa_handler(s);
	}
	// Mark the region as accessed
	(*i)->access++;
	mcontext_t *mCtx = &((ucontext_t *)ctx)->uc_mcontext;
	unsigned long writeAccess = mCtx->gregs[REG_ERR] & 0x2;
	if(!writeAccess) {
		TRACE("Read fault");
		(*i)->read(info->si_addr);
	}
	else {
		TRACE("Write fault");
		(*i)->write(info->si_addr);
	}
	TRACE("SIGSEGV done");
}

};

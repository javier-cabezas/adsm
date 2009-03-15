#include <common/MemRegion.h>
#include <common/debug.h>
#include <common/paraver.h>

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
	pushState(_gmacSignal_);
	bool isRegion = false;
	std::list<ProtRegion *>::iterator i;

	mcontext_t *mCtx = &((ucontext_t *)ctx)->uc_mcontext;
	unsigned long writeAccess = mCtx->gregs[REG_ERR] & 0x2;

	if(!writeAccess) TRACE("Read SIGSEGV for %p", info->si_addr);
	else TRACE("Write SIGSEGV for %p", info->si_addr);

	MUTEX_LOCK(regionMutex);
	i = std::find_if(regionList.begin(), regionList.end(),
		FindMem(info->si_addr));
	if(i != regionList.end()) isRegion = true;
	MUTEX_UNLOCK(regionMutex);

	if(isRegion && (*i)->isOwner() == false) isRegion = false;
	if(isRegion == false) {
		abort();
		// TODO: set the signal mask and other stuff
		if(defaultAction.sa_flags & SA_SIGINFO)
			return defaultAction.sa_sigaction(s, info, ctx);
		return defaultAction.sa_handler(s);
	}

	if(!writeAccess) (*i)->read(info->si_addr);
	else (*i)->write(info->si_addr);

	TRACE("SIGSEGV done");
	popState();
}

};

#include "time.h"
#include "debug.h"
#include <common/config.h>
#include <common/threads.h>

#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <values.h>
#include <signal.h>
#include <ucontext.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <errno.h>

#include <list>
#include <algorithm>

typedef struct {
	double time, max, min;
} stamp_t;

static const size_t iters = 8;
static const size_t bufferSize = 32 * 1024 * 1024;
static const size_t maxRegions = bufferSize / 4 / 1024;

static int nRegions = 0;
static int pageSize = 0;
static std::list<uint8_t *> regionList;
static MUTEX(regionMutex);
static struct sigaction defaultAction;
static unsigned nSignals = 0;

class FindMem {
protected:
	const uint8_t *addr;
public:
	FindMem(const uint8_t *addr) : addr(addr) {};
	bool operator()(const uint8_t *p) const {
		return addr >= p && addr < (p + (bufferSize / nRegions));
	}
};

static void memHandler(int s, siginfo_t *info, void *ctx)
{
	bool isRegion = false;
	std::list<uint8_t *>::const_iterator i;
	mcontext_t *mCtx = &((ucontext_t *)ctx)->uc_mcontext;
	unsigned long writeAccess = mCtx->gregs[REG_ERR] & 0x2;

	MUTEX_LOCK(regionMutex);
	i = std::find_if(regionList.begin(), regionList.end(),
		FindMem((uint8_t *)info->si_addr));
	if(i != regionList.end()) isRegion = true;
	MUTEX_UNLOCK(regionMutex);
	if(isRegion == false) { abort(); }
	
	if(!writeAccess) mprotect(info->si_addr, pageSize, PROT_READ);
	else mprotect(info->si_addr, pageSize, PROT_READ | PROT_WRITE);

	nSignals++;
}

void access(stamp_t *stamp, uint8_t *ptr, int prot = PROT_NONE)
{
	stamp->time = 0;
	stamp->min = MAXDOUBLE;
	stamp->max = MINDOUBLE;
	for(uint32_t n = 0; n < iters; n++) {
		// Setup the region list
		mprotect(ptr, bufferSize, prot);
		nSignals = 0;
		for(uint32_t i = 0; i < bufferSize; i += (bufferSize / nRegions)) {
			regionList.push_back(&ptr[i]);
		}

		uint32_t i = 0;
		usec_t start = getTime();
		for(; i < bufferSize; i += pageSize) {
			ptr[i] = 0;
		}
		usec_t end = getTime();

		stamp->time += end - start;
		stamp->max = MAX((end - start), stamp->max);
		stamp->min = MIN((end - start), stamp->min);
	}
	stamp->time = stamp->time / iters;
}

int main(int argc, char *argv[])
{
	struct sigaction segvAction;

	pageSize = getpagesize();	// Get protection granularity

	// Program signal handler
	memset(&segvAction, 0, sizeof(segvAction));
	segvAction.sa_sigaction = memHandler;
	segvAction.sa_flags = SA_SIGINFO | SA_RESTART;
	sigemptyset(&segvAction.sa_mask);
	if(sigaction(SIGSEGV, &segvAction, &defaultAction) < 0)
		FATAL();

	// Map a region of memory
	uint8_t *ptr = (uint8_t *)mmap(0, bufferSize, PROT_READ | PROT_WRITE,
		MAP_ANON | MAP_PRIVATE, 0, 0);
	if(ptr == MAP_FAILED) FATAL();


	size_t prevRegions = 0;
	double prevMetric = 0;
	fprintf(stdout,"#Regions\tTotal\tMin\tMax\tDelta\tAverage\tSignals\n");
	for(nRegions = 1; nRegions < (maxRegions + 1); nRegions += nRegions) {
		// Get reference time
		stamp_t stamp;
		access(&stamp, ptr, PROT_READ | PROT_WRITE);
		double reference = stamp.time;

		access(&stamp, ptr);

		size_t delta = nRegions - prevRegions;
		fprintf(stdout,"%d\t%f\t", nRegions, stamp.time / reference);
		fprintf(stdout,"%f\t%f\t", stamp.min / reference, stamp.max / reference);
		fprintf(stdout,"%f\t", ((stamp.time / reference) - prevMetric) / delta);
		fprintf(stdout,"%f\t%d\n", 1.0 * stamp.time / (bufferSize / nRegions) / reference, nSignals);

		prevMetric = stamp.time / reference;
		prevRegions = nRegions;
	}
}


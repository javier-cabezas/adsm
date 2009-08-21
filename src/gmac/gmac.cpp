#include <gmac.h>
#include <init.h>

#include <order.h>
#include <config.h>
#include <threads.h>
#include <debug.h>

#include <kernel/Process.h>
#include <kernel/Context.h>
#include <memory/MemManager.h>

#include <paraver.h>

#include <stdlib.h>
#include <assert.h>

MUTEX(gmacMutex);
static size_t pageSize = 0;

static const char *managerVar = "GMAC_MANAGER";

#ifdef PARAVER
namespace paraver {
extern int init;
}
#endif

static void __attribute__((constructor(CORE))) gmacInit(void)
{
	TRACE("Initialiazing GMAC");
#ifdef PARAVER
	paraver::init = 1;
#endif
	pageSize = getpagesize();
	MUTEX_INIT(gmacMutex);
	gmac::Process::init(getenv(managerVar));
	proc->context();
}


static void __attribute__((destructor)) gmacFini(void)
{
	TRACE("Cleaning GMAC");
	delete proc;
}

gmacError_t gmacMalloc(void **devPtr, size_t count)
{
	enterFunction(gmacMalloc);
	gmacError_t ret = gmacSuccess;
	count = (count < pageSize) ? pageSize : count;
	ret = gmac::Context::current()->malloc(devPtr, count);
	if(ret != gmacSuccess || !manager) {
		exitFunction();
		return ret;
	}
	if(!manager->alloc(*devPtr, count)) {
		gmac::Context::current()->free(*devPtr);
		exitFunction();
		return gmacErrorMemoryAllocation;
	}
	exitFunction();
	return gmacSuccess;
}

gmacError_t gmacSafeMalloc(void **cpuPtr, size_t count)
{
	enterFunction(gmacMalloc);
	gmacError_t ret = gmacSuccess;
	void *devPtr;
	count = (count < pageSize) ? pageSize : count;
	ret = gmac::Context::current()->malloc(&devPtr, count);
	if(ret != gmacSuccess || !manager) {
		exitFunction();
		return ret;
	}
	if((*cpuPtr = manager->safeAlloc(devPtr, count)) == NULL) {
		gmac::Context::current()->free(devPtr);
		exitFunction();
		return gmacErrorMemoryAllocation;
	}
	exitFunction();
	return gmacSuccess;
}

void *gmacSafePointer(void *devPtr)
{
	if(!manager) return devPtr;
	return manager->safe(devPtr);
}

gmacError_t gmacFree(void *devPtr)
{
	enterFunction(gmacFree);
	gmac::Context::current()->free(gmacSafePointer(devPtr));
	if(manager) {
		manager->release(devPtr);
	}
	exitFunction();
	return gmacSuccess;
}


gmacError_t gmacLaunch(const char *symbol)
{
	enterFunction(gmacLaunch);
	gmacError_t ret = gmacSuccess;
	if(manager) {
		TRACE("Memory Flush");
		manager->flush();
	}
	TRACE("Kernel Launch");
	ret = gmac::Context::current()->launch(symbol);
	ret = gmac::Context::current()->sync();
	exitFunction();
	return ret;
}

gmacError_t gmacThreadSynchronize()
{
	enterFunction(gmacSync);
	gmacError_t ret = gmac::Context::current()->sync();
	if(manager) {
		TRACE("Memory Sync");
		manager->sync();
	}
	exitFunction();
	return ret;
}

gmacError_t gmacGetLastError()
{
	return gmac::Context::current()->error();
}


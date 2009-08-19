#include <gmac.h>
#include <init.h>

#include <order.h>
#include <config.h>
#include <threads.h>
#include <debug.h>

#include <kernel/System.h>
#include <kernel/Context.h>
#include <memory/MemManager.h>

#include <paraver.h>

#include <stdlib.h>
#include <assert.h>

MUTEX(gmacMutex);
gmac::System *sys = NULL;
gmac::MemManager *memManager = NULL;
static unsigned memManagerCount = 0;
static size_t pageSize = 0;

static const char *memManagerVar = "GMAC_MANAGER";

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
	sys = new gmac::System();

	// Initialize subsystems
	gmac::apiInit();
	gmac::memoryInit();

	// Create initial memory manager and context
	gmac::createManager(getenv(memManagerVar));
	gmac::createContext();
}


static void __attribute__((destructor)) gmacFini(void)
{
	gmac::destroyManager();
}

gmacError_t gmacMalloc(void **devPtr, size_t count)
{
	enterFunction(gmacMalloc);
	gmacError_t ret = gmacSuccess;
	count = (count < pageSize) ? pageSize : count;
	ret = current->malloc(devPtr, count);
	if(ret != gmacSuccess || !memManager) {
		exitFunction();
		return ret;
	}
	if(!memManager->alloc(*devPtr, count)) {
		current->free(*devPtr);
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
	ret = current->malloc(&devPtr, count);
	if(ret != gmacSuccess || !memManager) {
		exitFunction();
		return ret;
	}
	if((*cpuPtr = memManager->safeAlloc(devPtr, count)) == NULL) {
		current->free(devPtr);
		exitFunction();
		return gmacErrorMemoryAllocation;
	}
	exitFunction();
	return gmacSuccess;
}

void *gmacSafePointer(void *devPtr)
{
	if(!memManager) return devPtr;
	return memManager->safe(devPtr);
}

gmacError_t gmacFree(void *devPtr)
{
	enterFunction(gmacFree);
	current->free(gmacSafePointer(devPtr));
	if(memManager) {
		memManager->release(devPtr);
	}
	exitFunction();
	return gmacSuccess;
}


gmacError_t gmacLaunch(const char *symbol)
{
	enterFunction(gmacLaunch);
	gmacError_t ret = gmacSuccess;
	if(memManager) {
		TRACE("Memory Flush");
		memManager->flush();
	}
	TRACE("Kernel Launch");
	ret = current->launch(symbol);
	exitFunction();
	return ret;
}

gmacError_t gmacThreadSynchronize()
{
	enterFunction(gmacSync);
	gmacError_t ret = current->sync();
	if(memManager) {
		TRACE("Memory Sync");
		memManager->sync();
	}
	exitFunction();
	return ret;
}

gmacError_t gmacGetLastError()
{
	return current->error();
}

const char *gmacGetErrorString(gmacError_t error)
{
	return NULL;
}

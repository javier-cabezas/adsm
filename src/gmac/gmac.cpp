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
	proc->create();
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

void *gmacMemset(void *s, int c, size_t n)
{
    void *ret = s;
	assert(manager != NULL);
	
	gmac::Context *ctx = manager->owner(s);
	assert(ctx != NULL);
	manager->invalidate(s, n);
	ctx->memset(manager->safe(s), c, n);

    return ret;
}

void *gmacMemcpy(void *dst, const void *src, size_t n)
{
	void *ret = dst;
	size_t ds = 0, ss = 0;

	assert(manager != NULL);

	// Locate memory regions (if any)
	gmac::Context *dstCtx = manager->owner(dst);
	gmac::Context *srcCtx = manager->owner(src);

	assert(dstCtx != NULL || srcCtx != NULL);

	TRACE("GMAC Memcpy");
	if(dstCtx == NULL) { // Copy to Host
		manager->flush(src, n);
		srcCtx->copyToHost(dst, manager->safe(src), n);
	}
	else if(srcCtx == NULL) { // Copy to Device
		manager->invalidate(dst, n);
		dstCtx->copyToDevice(manager->safe(dst), src, n);
	}
	else if(dstCtx == srcCtx) {	// Same device copy
		manager->flush(src, n);
		manager->invalidate(dst, n);
		dstCtx->copyDevice(manager->safe(dst),
				manager->safe(src), n);
	}
	else {
		void *tmp = malloc(n);
		manager->flush(src, n);
		srcCtx->copyToHost(tmp, manager->safe(src), n);
		manager->invalidate(dst, n);
		dstCtx->copyToDevice(manager->safe(dst), tmp, n);
		free(tmp);
	}

	return ret;

}

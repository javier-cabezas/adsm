#include <gmac.h>
#include <init.h>

#include <order.h>
#include <config.h>
#include <threads.h>
#include <debug.h>

#include <kernel/Process.h>
#include <kernel/Context.h>
#include <memory/Manager.h>

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

static gmacError_t __gmacMalloc(void **cpuPtr, size_t count)
{
	gmacError_t ret = gmacSuccess;
	void *devPtr;
	count = (count < pageSize) ? pageSize : count;
	ret = gmac::Context::current()->malloc(&devPtr, count);
	if(ret != gmacSuccess || !manager) {
		return ret;
	}
	if((*cpuPtr = manager->alloc(devPtr, count)) == NULL) {
		gmac::Context::current()->free(devPtr);
		return gmacErrorMemoryAllocation;
	}
	return gmacSuccess;
}

gmacError_t gmacMalloc(void **cpuPtr, size_t count)
{
	enterFunction(gmacMalloc);
	gmacError_t ret = __gmacMalloc(cpuPtr, count);
	exitFunction();
	return ret;
}

#ifdef USE_GLOBAL_HOST
gmacError_t gmacGlobalMalloc(void **cpuPtr, size_t count)
{
	enterFunction(gmacGlobalMalloc);
	gmacError_t ret = gmacSuccess;
	void *devPtr;
	count = (count < pageSize) ? pageSize : count;
	ret = gmac::Context::current()->hostMemAlign(cpuPtr, &devPtr, count);
	if(ret != gmacSuccess || !manager) {
		return ret;
	}
	proc->addShared(*cpuPtr, count);
	manager->map(*cpuPtr, devPtr, count);
	gmac::Process::ContextList::const_iterator i;
	for(i = proc->contexts().begin(); i != proc->contexts().end(); i++) {
		if(*i == gmac::Context::current()) continue;
		(*i)->hostMap(*cpuPtr, &devPtr, count);
		manager->remap(*i, *cpuPtr, devPtr, count);
	}
	exitFunction();
	return gmacSuccess;
}
#else
gmacError_t gmacGlobalMalloc(void **cpuPtr, size_t count)
{
	enterFunction(gmacGlobalMalloc);
	// Allocate memory in the current context
	gmacError_t ret = __gmacMalloc(cpuPtr, count);
	if(ret != gmacSuccess) {
		exitFunction();
		return ret;
	}
	// Comment this out if we opt for a hierarchy-based memory sharing
	void *devPtr;
	count = (count < pageSize) ? pageSize : count;
	proc->addShared(*cpuPtr, count);
	gmac::Process::ContextList::const_iterator i;
	for(i = proc->contexts().begin(); i != proc->contexts().end(); i++) {
		if(*i == gmac::Context::current()) continue;
		ret = (*i)->malloc(&devPtr, count);
		if(ret != gmacSuccess) goto cleanup;
		manager->remap(*i, *cpuPtr, devPtr, count);
	}
	exitFunction();
	return ret;

cleanup:
	gmac::Context *last = *i;
	for(i = proc->contexts().begin(); *i != last; i++) {
		(*i)->free(manager->ptr(*i, cpuPtr));
		manager->unmap(*i, *cpuPtr);
	}
	gmacFree(devPtr);
	exitFunction();
	return ret;
}
#endif

gmacError_t gmacFree(void *cpuPtr)
{
	enterFunction(gmacFree);
	if(manager) {
		manager->release(cpuPtr);
	}

	if(proc->isShared(cpuPtr)) {
		if(proc->removeShared(cpuPtr) == true)
			gmac::Context::current()->hostFree(cpuPtr);
	}
	else {
		gmac::Context::current()->free(cpuPtr);
	}
	exitFunction();
	return gmacSuccess;
}

void *gmacPtr(void *ptr)
{
	if(manager == NULL) return ptr;
	return manager->ptr(ptr);
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
	ctx->memset(manager->ptr(s), c, n);

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

    gmacError_t err;
	TRACE("GMAC Memcpy");
	if(dstCtx == NULL) { // Copy to Host
		manager->flush(src, n);
		err = srcCtx->copyToHost(dst, manager->ptr(srcCtx, src), n);
        assert(err == gmacSuccess);
	}
	else if(srcCtx == NULL) { // Copy to Device
		manager->invalidate(dst, n);
		err = dstCtx->copyToDevice(manager->ptr(dstCtx, dst), src, n);
        assert(err == gmacSuccess);
	}
	else if(dstCtx == srcCtx) {	// Same device copy
		manager->flush(src, n);
		manager->invalidate(dst, n);
		err = dstCtx->copyDevice(manager->ptr(dstCtx, dst),
			                     manager->ptr(dstCtx, src), n);
        assert(err == gmacSuccess);
	}
	else {
		void *tmp = malloc(n);
		manager->flush(src, n);
		err = srcCtx->copyToHost(tmp, manager->ptr(srcCtx, src), n);
        assert(err == gmacSuccess);
		manager->invalidate(dst, n);
		err = dstCtx->copyToDevice(manager->ptr(dstCtx, dst), tmp, n);
        assert(err == gmacSuccess);
		free(tmp);
	}

	return ret;

}

void gmacSendReceive(THREAD_ID id)
{
	gmac::Context *dst = proc->context(id);
	assert(dst != NULL);
	gmac::Context::current()->sendReceive(dst);
}

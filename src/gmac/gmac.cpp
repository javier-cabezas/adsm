#include <gmac.h>
#include <init.h>

#include <order.h>
#include <config.h>
#include <threads.h>
#include <debug.h>

#include <config/params.h>
#include <kernel/Process.h>
#include <kernel/Context.h>
#include <memory/Manager.h>

#include <paraver.h>

#include <cstdlib>
#include <cassert>

PRIVATE(__in_gmac);
const char __gmac_code = 1;
const char __user_code = 0;


#ifdef PARAVER
namespace paraver {
extern int init;
}
#endif

PARAM_REGISTER(paramMemManager,
	char *,
	"Rolling",
	"GMAC_MANAGER");
   
PARAM_REGISTER(paramPageSize,
	size_t,
	getpagesize(),
	NULL,
	PARAM_NONZERO);



static void __attribute__((constructor(CORE))) gmacInit(void)
{
	TRACE("Initialiazing GMAC");
#ifdef PARAVER
	paraver::init = 1;
#endif
	PRIVATE_INIT(__in_gmac, NULL);
	gmac::Process::init(paramMemManager);
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
	count = (count < paramPageSize) ? paramPageSize : count;
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
	__enterGmac();
	enterFunction(gmacMalloc);
	gmacError_t ret = __gmacMalloc(cpuPtr, count);
	exitFunction();
	__exitGmac();
	return ret;
}

#ifdef USE_GLOBAL_HOST
gmacError_t gmacGlobalMalloc(void **cpuPtr, size_t count)
{
	__enterGmac();
	enterFunction(gmacGlobalMalloc);
	gmacError_t ret = gmacSuccess;
	void *devPtr;
	count = (count < paramPageSize) ? paramPageSize : count;
	ret = gmac::Context::current()->hostMemAlign(cpuPtr, &devPtr, count);
	if(ret != gmacSuccess || !manager) {
		exitFunction();
		__exitGmac();
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
	__exitGmac();
	return gmacSuccess;
}
#else
gmacError_t gmacGlobalMalloc(void **cpuPtr, size_t count)
{
	__enterGmac();
	enterFunction(gmacGlobalMalloc);
	// Allocate memory in the current context
	gmacError_t ret = __gmacMalloc(cpuPtr, count);
	if(ret != gmacSuccess) {
		exitFunction();
		__exitGmac();
		return ret;
	}
	// Comment this out if we opt for a hierarchy-based memory sharing
	void *devPtr;
	count = (count < paramPageSize) ? paramPageSize : count;
	proc->addShared(*cpuPtr, count);
	gmac::Process::ContextList::const_iterator i;
	for(i = proc->contexts().begin(); i != proc->contexts().end(); i++) {
		if(*i == gmac::Context::current()) continue;
		ret = (*i)->malloc(&devPtr, count);
		if(ret != gmacSuccess) goto cleanup;
		manager->remap(*i, *cpuPtr, devPtr, count);
	}
	exitFunction();
	__exitGmac();
	return ret;

cleanup:
	gmac::Context *last = *i;
	for(i = proc->contexts().begin(); *i != last; i++) {
		(*i)->free(manager->ptr(*i, cpuPtr));
		manager->unmap(*i, *cpuPtr);
	}
	gmacFree(devPtr);
	exitFunction();
	__exitGmac();
	return ret;
}
#endif

gmacError_t gmacFree(void *cpuPtr)
{
	__enterGmac();
	enterFunction(gmacFree);
	if(manager) {
		manager->release(cpuPtr);
	}

	// If it is a shared global structure and nobody is accessing
	// it anymore, release the host memory
	if(proc->isShared(cpuPtr)) {
		if(proc->removeShared(cpuPtr) == true) {
			gmac::Context::current()->hostFree(cpuPtr);
		}
	}
	else {
		gmac::Context::current()->free(cpuPtr);
	}
	exitFunction();
	__exitGmac();
	return gmacSuccess;
}

void *gmacPtr(void *ptr)
{
	void *ret = NULL;
	__enterGmac();
	if(manager != NULL) ret = manager->ptr(ptr);
	__exitGmac();
	return ret;
}

gmacError_t gmacLaunch(const char *symbol)
{
	__enterGmac();
	enterFunction(gmacLaunch);
	gmacError_t ret = gmacSuccess;
	if(manager) {
		TRACE("Memory Flush");
		manager->flush();
	}
	TRACE("Kernel Launch");
	ret = gmac::Context::current()->launch(symbol);
	exitFunction();
	__exitGmac();
	return ret;
}

gmacError_t gmacThreadSynchronize()
{
	__enterGmac();
	enterFunction(gmacSync);
	gmacError_t ret = gmac::Context::current()->sync();
	if(manager) {
		TRACE("Memory Sync");
		manager->sync();
	}
	exitFunction();
	__exitGmac();
	return ret;
}

gmacError_t gmacGetLastError()
{
	__enterGmac();
	gmacError_t ret = gmac::Context::current()->error();
	__exitGmac();
	return ret;
}

void *gmacMemset(void *s, int c, size_t n)
{
	__enterGmac();
   void *ret = s;
	assert(manager != NULL);
	
	gmac::Context *ctx = manager->owner(s);
	assert(ctx != NULL);
	manager->invalidate(s, n);
	ctx->memset(manager->ptr(s), c, n);
	__exitGmac();
   return ret;
}

void *gmacMemcpy(void *dst, const void *src, size_t n)
{
	__enterGmac();
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
    // TODO - copyDevice can be always asynchronous
	else if(dstCtx == srcCtx) {	// Same device copy
		manager->flush(src, n);
		manager->invalidate(dst, n);
		err = dstCtx->copyDevice(manager->ptr(dstCtx, dst),
			                     manager->ptr(dstCtx, src), n);
        assert(err == gmacSuccess);
	}
    // TODO - add asynchronous calls to copyToHostAsync and copyToDeviceAsync
	else {
		void *tmp;
        bool pageLocked = false;

        manager->flush(src, n);
        manager->invalidate(dst, n);

        gmac::Context *ctx = gmac::Context::current();
        if (ctx->bufferPageLockedSize() > 0) {
            size_t bufferSize = ctx->bufferPageLockedSize();
            void * tmp        = ctx->bufferPageLocked();

            size_t left = n;
            off_t  off  = 0;
            while (left != 0) {
                size_t bytes = left < bufferSize? left: bufferSize;
                err = srcCtx->copyToHost(tmp, manager->ptr(srcCtx, ((char *) src) + off), bytes);
                assert(err == gmacSuccess);
                err = dstCtx->copyToDevice(manager->ptr(dstCtx, ((char *) dst) + off), tmp, bytes);
                assert(err == gmacSuccess);

                left -= bytes;
                off  += bytes;
                TRACE("Copying %zd %zd\n", bytes, bufferSize);
            }
        } else {
            TRACE("Allocated non-locked memory: %zd\n", n);
            tmp = malloc(n);

            err = srcCtx->copyToHost(tmp, manager->ptr(srcCtx, src), n);
            assert(err == gmacSuccess);
            err = dstCtx->copyToDevice(manager->ptr(dstCtx, dst), tmp, n);
            assert(err == gmacSuccess);
            free(tmp);
        }
	}
	__exitGmac();
	return ret;

}

void gmacSendReceive(unsigned long id)
{
	__enterGmac();
	proc->sendReceive(id);
	__exitGmac();
}

#include <gmac.h>
#include <init.h>

#include <order.h>
#include <config.h>
#include <threads.h>
#include <debug.h>

#include "util/Parameter.h"
#include "kernel/Process.h"
#include "kernel/Context.h"
#include "memory/Manager.h"

#include <paraver.h>

#include <cstdlib>

MUTEX(gmacMutex);

#ifdef PARAVER
namespace paraver {
extern int init;
}
#endif

PRIVATE(__in_gmac);

const char __gmac_code = 1;
const char __user_code = 0;

static void __attribute__((constructor))
gmacInit(void)
{
    TRACE("Initialiazing GMAC");
	PRIVATE_INIT(__in_gmac, NULL);
	__enterGmac();

#ifdef PARAVER
    paraver::init = 1;
    paraverInit();
#endif

    /* Call initialization of interpose libraries */
    osInit();
    threadInit();
    stdcInit();

    paramInit();

    TRACE("Using %s memory manager", paramMemManager);
    gmac::Process::init(paramMemManager);
    ASSERT(manager != NULL);
    __exitGmac();
}

static void __attribute__((destructor))
gmacFini(void)
{
	__enterGmac();
    TRACE("Cleaning GMAC");
    delete proc;
    __exitGmac();
}

gmacError_t
gmacClear(gmacKernel_t k)
{
    __enterGmac();
    enterFunction(gmacClear);
    gmac::Context * ctx = gmac::Context::current();
    gmac::Kernel  * kernel = ctx->kernel(k);

    if (kernel == NULL) {
        return gmacErrorInvalidValue;
    }
    kernel->clear();
    exitFunction();
    __exitGmac();
    return gmacSuccess;
}

gmacError_t
gmacBind(void * obj, gmacKernel_t k)
{
    __enterGmac();
    enterFunction(gmacBind);
    gmac::Context * ctx = gmac::Context::current();
    gmac::Kernel  * kernel = ctx->kernel(k);

    if (kernel == NULL) {
        return gmacErrorInvalidValue;
    }
    gmacError_t ret;
    ret = kernel->bind(obj);
    exitFunction();
    __exitGmac();
    return ret;
}

gmacError_t
gmacUnbind(void * obj, gmacKernel_t k)
{
    __enterGmac();
    enterFunction(gmacUnbind);
    gmac::Context * ctx = gmac::Context::current();
    gmac::Kernel  * kernel = ctx->kernel(k);

    if (kernel == NULL) {
        return gmacErrorInvalidValue;
    }
    gmacError_t ret;
    ret = kernel->unbind(obj);
	exitFunction();
	__exitGmac();
    return ret;
}

size_t
gmacAccs()
{
    size_t ret;
	__enterGmac();
	enterFunction(gmacAccs);
    ret = proc->accs();
	exitFunction();
	__exitGmac();
	return ret;
}

gmacError_t
gmacSetAffinity(int acc)
{
	gmacError_t ret;
	__enterGmac();
	enterFunction(gmacSetAffinity);
    ret = proc->migrate(acc);
	exitFunction();
	__exitGmac();
	return ret;
}

#if 0
static
gmacError_t __gmacMalloc(void **cpuPtr, size_t count)
{
	gmacError_t ret = gmacSuccess;
	void *devPtr;
	count = (count < getpagesize()) ? getpagesize(): count;
    gmac::Context * ctx = gmac::Context::current();
	ret = ctx->malloc(&devPtr, count);
	if(ret != gmacSuccess || !manager) {
		return ret;
	}
	if((*cpuPtr = manager->alloc(devPtr, count, attr)) == NULL) {
		ctx->free(devPtr);
		return gmacErrorMemoryAllocation;
	}
	return gmacSuccess;
}
#endif

gmacError_t
gmacMalloc(void **cpuPtr, size_t count)
{
	__enterGmac();
	enterFunction(gmacMalloc);
	count = (count < getpagesize()) ? getpagesize(): count;
	gmacError_t ret = manager->malloc(cpuPtr, count);
	exitFunction();
	__exitGmac();
	return ret;
}

gmacError_t
gmacGlobalMalloc(void **cpuPtr, size_t count)
{
#ifndef USE_MMAP
    __enterGmac();
    enterFunction(gmacGlobalMalloc);
	count = (count < getpagesize()) ? getpagesize(): count;
	gmacError_t ret = manager->globalMalloc(cpuPtr, count);
    exitFunction();
    __exitGmac();
    return ret;
#else
    return gmacErrorFeatureNotSupported;
#endif
}

gmacError_t
gmacFree(void *cpuPtr)
{
	__enterGmac();
	enterFunction(gmacFree);
	if(manager) {
		manager->free(cpuPtr);
	}

	// If it is a shared global structure and nobody is accessing
	// it anymore, release the host memory
	if(gmac::memory::Map::isShared(cpuPtr)) {
#ifdef USE_GLOBAL_HOST 
        gmac::Context::current()->hostFree(cpuPtr);
#endif
	}
	else {
		gmac::Context::current()->free(cpuPtr);
	}
	exitFunction();
	__exitGmac();
	return gmacSuccess;
}

void *
gmacPtr(void *ptr)
{
    void *ret = NULL;
    __enterGmac();
    ret = manager->ptr(ptr);
    __exitGmac();
    return ret;
}

gmacError_t
gmacLaunch(gmacKernel_t k)
{
    __enterGmac();
    enterFunction(gmacLaunch);
    gmac::Context * ctx = gmac::Context::current();
    gmac::KernelLaunch * launch = ctx->launch(k);

    gmacError_t ret = gmacSuccess;
    TRACE("Flush the memory used in the kernel");
    manager->flush(*launch);

    TRACE("Kernel Launch");
    ret = launch->execute();

#if 0
    // Now automatically detected in the memory Handler
    if (paramAcquireOnRead) {
        ret = ctx->sync();
    }
#endif

    if(paramAcquireOnWrite) {
        TRACE("Invalidate the memory used in the kernel");
        manager->invalidate(*launch);
    }

    exitFunction();
    __exitGmac();

    return ret;
}

gmacError_t
gmacThreadSynchronize()
{
	__enterGmac();
	enterFunction(gmacSync);
    gmac::Context * ctx = gmac::Context::current();
	gmacError_t ret = ctx->sync();

    TRACE("Memory Sync");
    manager->invalidate(ctx->releaseRegions());

	exitFunction();
	__exitGmac();
	return ret;
}

gmacError_t
gmacGetLastError()
{
	__enterGmac();
	gmacError_t ret = gmac::Context::current()->error();
	__exitGmac();
	return ret;
}

void *
gmacMemset(void *s, int c, size_t n)
{
    __enterGmac();
    void *ret = s;
    gmac::Context *ctx = manager->owner(s);
    ASSERT(ctx != NULL);
    manager->invalidate(s, n);
    ctx->memset(manager->ptr(s), c, n);
	__exitGmac();
    return ret;
}

void *
gmacMemcpy(void *dst, const void *src, size_t n)
{
	__enterGmac();
	void *ret = dst;
	size_t ds = 0, ss = 0;

    gmacError_t err;
#if 0
    err = gmacThreadSynchronize();
    if (err != gmacSuccess) return NULL;
#endif

	// Locate memory regions (if any)
	gmac::Context *dstCtx = manager->owner(dst);
	gmac::Context *srcCtx = manager->owner(src);

	if (dstCtx != NULL && srcCtx != NULL) return NULL;

    if (srcCtx->status() == gmac::Context::RUNNING) srcCtx->sync();

    // TODO - copyDevice can be always asynchronous
	if(dstCtx == srcCtx) {	// Same device copy
		manager->flush(src, n);
		manager->invalidate(dst, n);
		err = dstCtx->copyDevice(manager->ptr(dstCtx, dst),
			                     manager->ptr(dstCtx, src), n);
        ASSERT(err == gmacSuccess);
	}
	else { // dstCtx != srcCtx
		void *tmp;
        bool pageLocked = false;

        gmac::Context *ctx = gmac::Context::current();

        manager->flush(src, n);
        manager->invalidate(dst, n);

        if (srcCtx->async() && dstCtx->async()) {
            size_t bufferSize = ctx->bufferPageLockedSize();
            void * tmp        = ctx->bufferPageLocked();

            size_t left = n;
            off_t  off  = 0;
            while (left != 0) {
                size_t bytes = left < bufferSize? left: bufferSize;
                err = srcCtx->copyToHostAsync(tmp, manager->ptr(srcCtx, ((char *) src) + off), bytes);
                ASSERT(err == gmacSuccess);
                srcCtx->syncToHost();
                ASSERT(err == gmacSuccess);
                err = dstCtx->copyToDeviceAsync(manager->ptr(dstCtx, ((char *) dst) + off), tmp, bytes);
                ASSERT(err == gmacSuccess);
                srcCtx->syncToDevice();
                ASSERT(err == gmacSuccess);

                left -= bytes;
                off  += bytes;
                TRACE("Copying %zd %zd\n", bytes, bufferSize);
            }
        } else {
            TRACE("Allocated non-locked memory: %zd\n", n);
            tmp = malloc(n);

            err = srcCtx->copyToHost(tmp, manager->ptr(srcCtx, src), n);
            ASSERT(err == gmacSuccess);
            err = dstCtx->copyToDevice(manager->ptr(dstCtx, dst), tmp, n);
            ASSERT(err == gmacSuccess);
            free(tmp);
        }
	}
	__exitGmac();
	return ret;

}

void
gmacSendReceive(unsigned long id)
{
	__enterGmac();
	proc->sendReceive((THREAD_ID)id);
	__exitGmac();
}

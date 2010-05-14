#include <gmac.h>
#include <init.h>

#include <order.h>
#include <config.h>
#include <threads.h>
#include <debug.h>

#include <util/Parameter.h>
#include <util/Private.h>
#include <util/Logger.h>

#include <kernel/Process.h>
#include <kernel/Context.h>

#include <memory/Manager.h>
#include <memory/Allocator.h>

#include <paraver.h>

#include <cstdlib>

#ifdef PARAVER
namespace paraver {
extern int init;
}
#endif

gmac::util::Logger *logger = NULL;
gmac::util::Private __in_gmac;

const char __gmac_code = 1;
const char __user_code = 0;

static void __attribute__((constructor))
gmacInit(void)
{
	gmac::util::Private::init(__in_gmac);
	__enterGmac();

    logger = new gmac::util::Logger("GMAC");
    logger->trace("Initialiazing GMAC");

#ifdef PARAVER
    paraver::init = 1;
    paraverInit();
#endif

    paramInit();

    /* Call initialization of interpose libraries */
    osInit();
    threadInit();
    stdcInit();

    logger->trace("Using %s memory manager", paramMemManager);
    logger->trace("Using %s memory allocator", paramMemAllocator);
    gmac::Process::init(paramMemManager, paramMemAllocator);
    logger->assertion(manager != NULL);
    __exitGmac();
}

static void __attribute__((destructor))
gmacFini(void)
{
	__enterGmac();
    logger->trace("Cleaning GMAC");
    delete proc;
    // We do not exitGmac to allow proper stdc function handling
    delete logger;
}

gmacError_t
gmacClear(gmacKernel_t k)
{
    __enterGmac();
    enterFunction(FuncGmacClear);
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
    enterFunction(FuncGmacBind);
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
    enterFunction(FuncGmacUnbind);
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
	enterFunction(FuncGmacAccs);
    ret = proc->nAccelerators();
	exitFunction();
	__exitGmac();
	return ret;
}

gmacError_t
gmacSetAffinity(int acc)
{
	gmacError_t ret;
	__enterGmac();
	enterFunction(FuncGmacSetAffinity);
    gmac::Context * ctx = gmac::Context::current();
    // We are potentially modifying the context, let's lock it
    ctx->lockWrite();
    ret = proc->migrate(ctx, acc);
    ctx->unlock();
	exitFunction();
	__exitGmac();
	return ret;
}

gmacError_t
gmacMalloc(void **cpuPtr, size_t count)
{
    gmacError_t ret = gmacSuccess;
	__enterGmac();
	enterFunction(FuncGmacMalloc);
    gmac::Context * ctx = gmac::Context::current();
    ctx->lockRead();
    if(allocator != NULL && count < (paramPageSize / 2)) {
        *cpuPtr = allocator->alloc(count, __builtin_return_address(0));   
    }
    else {
	    count = (int(count) < getpagesize())? getpagesize(): count;
	    ret = manager->malloc(ctx, cpuPtr, count);
    }
    ctx->unlock();
	exitFunction();
	__exitGmac();
	return ret;
}

gmacError_t
gmacGlobalMalloc(void **cpuPtr, size_t count)
{
#ifndef USE_MMAP
    __enterGmac();
    enterFunction(FuncGmacGlobalMalloc);
	count = (count < (size_t)getpagesize()) ? (size_t)getpagesize(): count;
    gmac::Context * ctx = gmac::Context::current();
    ctx->lockRead();
	gmacError_t ret = manager->globalMalloc(ctx, cpuPtr, count);
    ctx->unlock();
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
    gmacError_t ret = gmacSuccess;
	__enterGmac();
	enterFunction(FuncGmacFree);
    gmac::Context * ctx = manager->owner(cpuPtr);
    ctx->lockRead();
    if(allocator == NULL || allocator->free(cpuPtr) == false)
        ret = manager->free(ctx, cpuPtr);
    ctx->unlock();
	exitFunction();
	__exitGmac();
	return ret;
}

void *
gmacPtr(void *ptr)
{
    void *ret = NULL;
    __enterGmac();
    gmac::Context * ctx = gmac::Context::current();
    ctx->lockRead();
    ret = manager->ptr(ctx, ptr);
    ctx->unlock();
    __exitGmac();
    return ret;
}

gmacError_t
gmacLaunch(gmacKernel_t k)
{
    __enterGmac();
    enterFunction(FuncGmacLaunch);
    gmac::Context * ctx = gmac::Context::current();
    ctx->lockRead();
    gmac::KernelLaunch * launch = ctx->launch(k);

    gmacError_t ret = gmacSuccess;
    logger->trace("Flush the memory used in the kernel");
    manager->flush(*launch);

    // Wait for pending transfers
    ctx->syncToDevice();
    logger->trace("Kernel Launch");
    ret = launch->execute();

#if 0
    // Now automatically detected in the memory Handler
    if (paramAcquireOnRead) {
        ret = ctx->sync();
    }
#endif

    if(paramAcquireOnWrite) {
        logger->trace("Invalidate the memory used in the kernel");
        manager->invalidate(*launch);
    }

    ctx->unlock();
    exitFunction();
    __exitGmac();

    return ret;
}

gmacError_t
gmacThreadSynchronize()
{
	__enterGmac();
	enterFunction(FuncGmacSync);
    gmac::Context * ctx = gmac::Context::current();
    ctx->lockRead();
	gmacError_t ret = ctx->sync();

    logger->trace("Memory Sync");
    manager->invalidate(ctx->releaseRegions());

    ctx->unlock();
	exitFunction();
	__exitGmac();
	return ret;
}

gmacError_t
gmacGetLastError()
{
	__enterGmac();
    gmac::Context * ctx = gmac::Context::current();
    ctx->lockRead();
	gmacError_t ret = ctx->error();
    ctx->unlock();
	__exitGmac();
	return ret;
}

void *
gmacMemset(void *s, int c, size_t n)
{
    __enterGmac();
    void *ret = s;
    gmac::Context *ctx = manager->owner(s);
    ctx->lockRead();
    logger->cfatal(ctx != NULL, "No owner for %p\n", s);
    manager->invalidate(s, n);
    ctx->memset(manager->ptr(ctx, s), c, n);
    ctx->unlock();
	__exitGmac();
    return ret;
}

void *
gmacMemcpy(void *dst, const void *src, size_t n)
{
	__enterGmac();
	void *ret = dst;

    gmacError_t err;
#if 0
    err = gmacThreadSynchronize();
    if (err != gmacSuccess) return NULL;
#endif

	// Locate memory regions (if any)
	gmac::Context *dstCtx = manager->owner(dst);
	gmac::Context *srcCtx = manager->owner(src);

	if (dstCtx == NULL && srcCtx == NULL) return NULL;

	if (srcCtx != NULL) srcCtx->lockRead();
	if (dstCtx != NULL) dstCtx->lockRead();

    // TODO - copyDevice can be always asynchronous
	if(dstCtx == NULL) {	    // From device
		manager->flush(src, n);
		err = srcCtx->copyToHost(dst,
			                     manager->ptr(srcCtx, src), n);
        logger->assertion(err == gmacSuccess);
	}
    else if(srcCtx == NULL) {   // To device
		manager->invalidate(dst, n);
		err = dstCtx->copyToDevice(manager->ptr(dstCtx, dst),
			                       src, n);
        logger->assertion(err == gmacSuccess);
    }
    else if(dstCtx == srcCtx) {	// Same device copy
		manager->flush(src, n);
		manager->invalidate(dst, n);
		err = dstCtx->copyDevice(manager->ptr(dstCtx, dst),
			                     manager->ptr(srcCtx, src), n);
        logger->assertion(err == gmacSuccess);
	}
	else { // dstCtx != srcCtx
		//void *tmp;
        gmac::Context *ctx = gmac::Context::current();
        ctx->lockRead();

        manager->flush(src, n);
        manager->invalidate(dst, n);

        size_t bufferSize = ctx->bufferPageLockedSize();
        void * tmp        = ctx->bufferPageLocked();

        size_t left = n;
        off_t  off  = 0;
        while (left != 0) {
            size_t bytes = left < bufferSize? left: bufferSize;
            err = srcCtx->copyToHostAsync(tmp, manager->ptr(srcCtx, ((char *) src) + off), bytes);
            logger->assertion(err == gmacSuccess);
            srcCtx->syncToHost();
            logger->assertion(err == gmacSuccess);
            err = dstCtx->copyToDeviceAsync(manager->ptr(dstCtx, ((char *) dst) + off), tmp, bytes);
            logger->assertion(err == gmacSuccess);
            srcCtx->syncToDevice();
            logger->assertion(err == gmacSuccess);

            left -= bytes;
            off  += bytes;
            logger->trace("Copying %zd %zd\n", bytes, bufferSize);
        }
        ctx->unlock();
	}

	if (srcCtx != NULL) srcCtx->unlock();
	if (dstCtx != NULL) dstCtx->unlock();

	__exitGmac();
	return ret;

}

void
gmacSend(pthread_t id)
{
    __enterGmac();
    proc->send((THREAD_ID)id);
    __exitGmac();
}

void gmacReceive()
{
    __enterGmac();
    proc->receive();
    __exitGmac();
}

void
gmacSendReceive(pthread_t id)
{
	__enterGmac();
	proc->sendReceive((THREAD_ID)id);
	__exitGmac();
}

void gmacCopy(pthread_t id)
{
    __enterGmac();
    proc->copy((THREAD_ID)id);
    __exitGmac();
}

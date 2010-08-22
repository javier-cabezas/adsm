#include <gmac.h>
#include <init.h>

#include <order.h>
#include <config.h>
#include <threads.h>

#include <util/Parameter.h>
#include <util/Private.h>
#include <util/Logger.h>
#include <util/FileLock.h>

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

gmac::util::Private __in_gmac;

const char __gmac_code = 1;
const char __user_code = 0;

char __gmac_init = 0;

#ifdef LINUX 
#define GLOBAL_FILE_LOCK "/tmp/gmacSystemLock"
#else
#ifdef DARWIN
#define GLOBAL_FILE_LOCK "/tmp/gmacSystemLock"
#endif
#endif

static void __attribute__((constructor))
gmacInit(void)
{
	gmac::util::Private::init(__in_gmac);
	__enterGmac();
    __gmac_init = 1;

    gmac::util::Logger::Create("GMAC");
    gmac::util::Logger::TRACE("Initialiazing GMAC");

#ifdef PARAVER
    paraver::init = 1;
    paraverInit();
#endif
    //gmac::util::FileLock(GLOBAL_FILE_LOCK, paraver::LockSystem);

    //FILE * lockSystem;

    paramInit();

    /* Call initialization of interpose libraries */
    osInit();
    threadInit();
    stdcInit();

    gmac::util::Logger::TRACE("Using %s memory manager", paramProtocol);
    gmac::util::Logger::TRACE("Using %s memory allocator", paramAllocator);
    gmac::Process::init(paramProtocol, paramAllocator);
    gmac::util::Logger::ASSERTION(manager != NULL);
    __exitGmac();
}

static void __attribute__((destructor))
gmacFini(void)
{
	__enterGmac();
    gmac::util::Logger::TRACE("Cleaning GMAC");
    delete proc;
    // We do not exitGmac to allow proper stdc function handling
    gmac::util::Logger::Destroy();
}

gmacError_t
gmacClear(gmacKernel_t k)
{
    __enterGmac();
    enterFunction(FuncGmacClear);
    gmac::Mode * mode = gmac::Mode::current();
    gmac::util::Logger::ASSERTION(mode != NULL);
    gmac::Kernel  * kernel = mode->context().kernel(k);

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
    gmac::Mode *mode = gmac::Mode::current();
    gmac::util::Logger::ASSERTION(mode != NULL);
    gmac::Kernel  * kernel = mode->context().kernel(k);

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
    gmac::Mode * mode = gmac::Mode::current();
    gmac::util::Logger::ASSERTION(mode != NULL);
    gmac::Kernel  * kernel = mode->context().kernel(k);

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
    if (gmac::Mode::hasCurrent()) {
        gmac::Mode *mode = gmac::Mode::current();
        gmac::util::Logger::ASSERTION(mode != NULL);
        gmac::Context &ctx = mode->context();
        // We are potentially modifying the context, let's lock it
        ctx.lockWrite();
        ret = proc->migrate(mode, acc);
        ctx.unlock();
    } else {
        ret = proc->migrate(NULL, acc);
    }
	exitFunction();
	__exitGmac();
	return ret;
}

gmacError_t
gmacMalloc(void **cpuPtr, size_t count)
{
    gmacError_t ret = gmacSuccess;
    if (count == 0) {
        *cpuPtr = NULL;
        return ret;
    }
	__enterGmac();
	enterFunction(FuncGmacMalloc);
    if(allocator != NULL && count < (paramPageSize / 2)) {
        *cpuPtr = allocator->alloc(count, __builtin_return_address(0));   
    }
    else {
	    count = (int(count) < getpagesize())? getpagesize(): count;
	    ret = manager->alloc(cpuPtr, count);
    }
	exitFunction();
	__exitGmac();
	return ret;
}

gmacError_t
gmacGlobalMalloc(void **cpuPtr, size_t count)
{
#ifndef USE_MMAP
    gmacError_t ret = gmacSuccess;
    if(count == 0) {
        *cpuPtr = NULL;
        return ret;
    }
    __enterGmac();
    enterFunction(FuncGmacGlobalMalloc);
	count = (count < (size_t)getpagesize()) ? (size_t)getpagesize(): count;
    gmac::Context * ctx = gmac::Context::current();
    ctx->lockRead();
	ret = manager->globalMalloc(ctx, cpuPtr, count);
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
    if(allocator == NULL || allocator->free(cpuPtr) == false)
        ret = manager->free(cpuPtr);
	exitFunction();
	__exitGmac();
	return ret;
}

void *
gmacPtr(void *ptr)
{
    void *ret = NULL;
    __enterGmac();
    gmac::Mode *mode = gmac::Mode::current();
    ret = mode->translate(ptr);
    __exitGmac();
    return ret;
}

gmacError_t
gmacLaunch(gmacKernel_t k)
{
    __enterGmac();
    enterFunction(FuncGmacLaunch);
    gmac::Mode *mode = gmac::Mode::current();
    gmac::util::Logger::ASSERTION(mode != NULL);
    gmac::Context &ctx = mode->context();
    ctx.lockRead();
    gmac::KernelLaunch * launch = ctx.launch(k);

    gmacError_t ret = gmacSuccess;
    gmac::util::Logger::TRACE("Flush the memory used in the kernel");
    manager->release();

    // Wait for pending transfers
    ctx.syncToDevice();
    gmac::util::Logger::TRACE("Kernel Launch");
    ret = launch->execute();

#if 0
    // Now automatically detected in the memory Handler
    if (paramAcquireOnRead) {
        ret = ctx->sync();
    }
#endif

    if(paramAcquireOnWrite) {
        gmac::util::Logger::TRACE("Invalidate the memory used in the kernel");
        manager->invalidate();
    }

    ctx.unlock();
    exitFunction();
    __exitGmac();

    return ret;
}

gmacError_t
gmacThreadSynchronize()
{
	__enterGmac();
	enterFunction(FuncGmacSync);
    gmac::Mode *mode = gmac::Mode::current();
    gmac::util::Logger::ASSERTION(mode != NULL);
    gmac::Context &ctx = mode->context();
    ctx.lockRead();
	gmacError_t ret = ctx.sync();

    gmac::util::Logger::TRACE("Memory Sync");
    manager->invalidate();

    ctx.unlock();
	exitFunction();
	__exitGmac();
	return ret;
}

gmacError_t
gmacGetLastError()
{
	__enterGmac();
    gmac::Mode *mode = gmac::Mode::current();
    gmac::util::Logger::ASSERTION(mode != NULL);
    gmac::Context &ctx = mode->context();
    ctx.lockRead();
	gmacError_t ret = ctx.error();
    ctx.unlock();
	__exitGmac();
	return ret;
}

void *
gmacMemset(void *s, int c, size_t n)
{
    __enterGmac();
    void *ret = s;
    gmac::Mode *mode = proc->owner(s);
    gmac::util::Logger::ASSERTION(mode != NULL);
    gmac::Context &ctx = mode->context();
    ctx.lockRead();
    manager->invalidate(s, n);
    ctx.memset(proc->translate(s), c, n);
    ctx.unlock();
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
    gmac::Mode *dstMode = proc->owner(dst);
    gmac::Mode *srcMode = proc->owner(src);
	if (dstMode == NULL && srcMode == NULL) return NULL;

	gmac::Context &dstCtx = dstMode->context();
	gmac::Context &srcCtx = srcMode->context();

    // TODO - copyDevice can be always asynchronous
	if(dstMode == NULL) {	    // From device
        srcCtx.lockRead();
		manager->release((void *)src, n);
		err = srcCtx.copyToHost(dst, proc->translate(src), n);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
        srcCtx.unlock();
	}
    else if(srcMode == NULL) {   // To device
        dstCtx.lockRead();
		manager->invalidate(dst, n);
		err = dstCtx.copyToDevice(proc->translate(dst),src, n);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
        dstCtx.unlock();
    }
    else if(&dstCtx == &srcCtx) {	// Same device copy
        dstCtx.lockRead();
		manager->release((void *)src, n);
		manager->invalidate(dst, n);
		err = dstCtx.copyDevice(proc->translate(dst), proc->translate(src), n);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
        dstCtx.unlock();
	}
	else { // dstCtx != srcCtx
        gmac::Mode *mode = gmac::Mode::current();
        gmac::util::Logger::ASSERTION(mode != NULL);
        gmac::Context &ctx = mode->context();
        if(&ctx != &srcCtx) srcCtx.lockRead();
        if(&ctx != &dstCtx) dstCtx.lockRead();

        manager->release((void *)src, n);
        manager->invalidate(dst, n);

        size_t bufferSize = ctx.bufferPageLockedSize();
        void * tmp        = ctx.bufferPageLocked();

        size_t left = n;
        off_t  off  = 0;
        while (left != 0) {
            size_t bytes = left < bufferSize ? left : bufferSize;
            err = srcCtx.copyToHostAsync(tmp, proc->translate(((char *) src) + off), bytes);
            gmac::util::Logger::ASSERTION(err == gmacSuccess);
            srcCtx.syncToHost();
            gmac::util::Logger::ASSERTION(err == gmacSuccess);
            err = dstCtx.copyToDeviceAsync(proc->translate(((char *) dst) + off), tmp, bytes);
            gmac::util::Logger::ASSERTION(err == gmacSuccess);
            srcCtx.syncToDevice();
            gmac::util::Logger::ASSERTION(err == gmacSuccess);

            left -= bytes;
            off  += bytes;
            gmac::util::Logger::TRACE("Copying %zd %zd\n", bytes, bufferSize);
        }

        if(&ctx != &srcCtx) srcCtx.unlock();
        if(&ctx != &dstCtx) dstCtx.unlock();
        ctx.unlock();
	}

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

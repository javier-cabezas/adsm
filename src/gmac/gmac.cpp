#include <gmac.h>
#include <init.h>

#include <order.h>
#include <config.h>
#include <threads.h>

#include <util/Parameter.h>
#include <util/Private.h>
#include <util/Logger.h>
#include <util/FileLock.h>
#include <util/Function.h>

#include <kernel/Process.h>
#include <kernel/Context.h>
#include <kernel/IOBuffer.h>

#include <memory/Manager.h>
#include <memory/Allocator.h>

#include <config/paraver.h>

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


#if 0
gmacError_t
gmacClear(gmacKernel_t k)
{
    gmacError_t ret = gmacSuccess;
    __enterGmac();
    enterFunction(FuncGmacClear);
    gmac::Kernel *kernel = gmac::Mode::current()->kernel(k);
    if (kernel == NULL) ret = gmacErrorInvalidValue;
    else kernel->clear();
    exitFunction();
    __exitGmac();
    return ret;
}

gmacError_t
gmacBind(void * obj, gmacKernel_t k)
{
    gmacError_t ret = gmacSuccess;
    __enterGmac();
    enterFunction(FuncGmacBind);
    gmac::Kernel *kernel = gmac::Mode::current()->kernel(k);

    if (kernel == NULL) ret = gmacErrorInvalidValue;
    else ret = kernel->bind(obj);
    exitFunction();
    __exitGmac();
    return ret;
}

gmacError_t
gmacUnbind(void * obj, gmacKernel_t k)
{
    gmacError_t ret = gmacSuccess;
    __enterGmac();
    enterFunction(FuncGmacUnbind);
    gmac::Kernel  * kernel = gmac::Mode::current()->kernel(k);
    if (kernel == NULL) ret = gmacErrorInvalidValue;
    else ret = kernel->unbind(obj);
	exitFunction();
	__exitGmac();
    return ret;
}
#endif

size_t
gmacAccs()
{
    size_t ret;
	__enterGmac();
    gmac::util::Function::start("gmacAccs");
    ret = proc->nAccelerators();
    gmac::util::Function::end();
	__exitGmac();
	return ret;
}

gmacError_t
gmacSetAffinity(int acc)
{
	gmacError_t ret;
	__enterGmac();
    gmac::util::Function::start("gmacSetAffinity");
    if (gmac::Mode::hasCurrent()) {
        ret = proc->migrate(gmac::Mode::current(), acc);
    } else {
        ret = proc->migrate(NULL, acc);
    }
    gmac::util::Function::end();
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
    gmac::util::Function::start("gmacMalloc");
    if(allocator != NULL && count < (paramPageSize / 2)) {
        *cpuPtr = allocator->alloc(count, __builtin_return_address(0));   
    }
    else {
	    count = (int(count) < getpagesize())? getpagesize(): count;
	    ret = manager->alloc(cpuPtr, count);
    }
    gmac::util::Function::end();
	__exitGmac();
	return ret;
}

gmacError_t
gmacGlobalMalloc(void **cpuPtr, size_t count, int hint)
{
#ifndef USE_MMAP
    gmacError_t ret = gmacSuccess;
    if(count == 0) {
        *cpuPtr = NULL;
        return ret;
    }
    __enterGmac();
    gmac::util::Function::start("gmacGlobalMalloc");
	count = (count < (size_t)getpagesize()) ? (size_t)getpagesize(): count;
	ret = manager->globalAlloc(cpuPtr, count, hint);
    gmac::util::Function::end();
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
    gmac::util::Function::start("gmacFree");
    if(allocator == NULL || allocator->free(cpuPtr) == false)
        ret = manager->free(cpuPtr);
    gmac::util::Function::end();
	__exitGmac();
	return ret;
}

void *
gmacPtr(void *ptr)
{
    void *ret = NULL;
    __enterGmac();
    ret = proc->translate(ptr);
    __exitGmac();
    return ret;
}

gmacError_t
gmacLaunch(gmacKernel_t k)
{
    __enterGmac();
    gmac::util::Function::start("gmacLaunch");
    gmac::KernelLaunch * launch = gmac::Mode::current()->launch(k);

    gmacError_t ret = gmacSuccess;
    gmac::util::Logger::TRACE("Flush the memory used in the kernel");
    gmac::util::Logger::ASSERTION(manager->release() == gmacSuccess);

    // Wait for pending transfers
    gmac::Mode::current()->sync();
    gmac::util::Logger::TRACE("Kernel Launch");
    ret = launch->execute();

    if(paramAcquireOnWrite) {
        gmac::util::Logger::TRACE("Invalidate the memory used in the kernel");
        manager->invalidate();
    }

    delete launch;
    gmac::util::Function::end();
    __exitGmac();

    return ret;
}

gmacError_t
gmacThreadSynchronize()
{
	__enterGmac();
    gmac::util::Function::start("gmacSync");

	gmacError_t ret = gmac::Mode::current()->sync();
    gmac::util::Logger::TRACE("Memory Sync");
    manager->acquire();

    gmac::util::Function::end();
	__exitGmac();
	return ret;
}

gmacError_t
gmacGetLastError()
{
	__enterGmac();
	gmacError_t ret = gmac::Mode::current()->error();
	__exitGmac();
	return ret;
}

void *
gmacMemset(void *s, int c, size_t n)
{
    __enterGmac();
    void *ret = s;
    gmac::Mode *mode = gmac::Mode::current();
    manager->invalidate(s, n);
    mode->memset(proc->translate(s), c, n);
	__exitGmac();
    return ret;
}


void *
gmacMemcpy(void *dst, const void *src, size_t n)
{
	__enterGmac();
	void *ret = dst;

    gmacError_t err;

	// Locate memory regions (if any)
    gmac::Mode *dstMode = proc->owner(dst);
    gmac::Mode *srcMode = proc->owner(src);
	if (dstMode == NULL && srcMode == NULL) return NULL;

    // TODO - copyDevice can be always asynchronous
	if(dstMode == NULL) {	    // From device
		manager->release((void *)src, n);
		err = srcMode->copyToHost(dst, proc->translate(src), n);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
	}
    else if(srcMode == NULL) {   // To device
		manager->invalidate(dst, n);
		err = dstMode->copyToDevice(proc->translate(dst),src, n);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
    }
    else if(dstMode == srcMode) {	// Same device copy
		manager->release((void *)src, n);
		manager->invalidate(dst, n);
		err = dstMode->copyDevice(proc->translate(dst), proc->translate(src), n);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
	}
	else { // dstCtx != srcCtx
        gmac::Mode *mode = gmac::Mode::current();
        gmac::util::Logger::ASSERTION(mode != NULL);

        manager->release((void *)src, n);
        manager->invalidate(dst, n);

        off_t off = 0;
        gmac::IOBuffer *buffer = gmac::Mode::current()->getIOBuffer();

        size_t left = n;
        while (left != 0) {
            size_t bytes = left < buffer->size() ? left : buffer->size();
            err = srcMode->bufferToHost(buffer, proc->translate((char *)src + off), bytes);
            gmac::util::Logger::ASSERTION(err == gmacSuccess);

            err = dstMode->bufferToDevice(buffer, proc->translate((char *)dst + off), bytes);
            gmac::util::Logger::ASSERTION(err == gmacSuccess);

            left -= bytes;
            off  += bytes;
        }

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

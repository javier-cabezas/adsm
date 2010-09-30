#include <gmac.h>
#include "init.h"

#include <order.h>
#include <config.h>
#include <threads.h>

#include "util/Logger.h"

#include "core/Context.h"
#include "core/Mode.h"
#include "core/IOBuffer.h"
#include "core/Process.h"

#include "memory/Manager.h"
#include "memory/Allocator.h"

#include "trace/Function.h"

#include <cstdlib>

#if 0
gmacError_t
gmacClear(gmacKernel_t k)
{
    gmacError_t ret = gmacSuccess;
    gmac::enterGmac();
    enterFunction(FuncGmacClear);
    gmac::Kernel *kernel = gmac::Mode::current()->kernel(k);
    if (kernel == NULL) ret = gmacErrorInvalidValue;
    else kernel->clear();
    exitFunction();
    gmac::exitGmac();
    return ret;
}

gmacError_t
gmacBind(void * obj, gmacKernel_t k)
{
    gmacError_t ret = gmacSuccess;
    gmac::enterGmac();
    enterFunction(FuncGmacBind);
    gmac::Kernel *kernel = gmac::Mode::current()->kernel(k);

    if (kernel == NULL) ret = gmacErrorInvalidValue;
    else ret = kernel->bind(obj);
    exitFunction();
    gmac::exitGmac();
    return ret;
}

gmacError_t
gmacUnbind(void * obj, gmacKernel_t k)
{
    gmacError_t ret = gmacSuccess;
    gmac::enterGmac();
    enterFunction(FuncGmacUnbind);
    gmac::Kernel  * kernel = gmac::Mode::current()->kernel(k);
    if (kernel == NULL) ret = gmacErrorInvalidValue;
    else ret = kernel->unbind(obj);
	exitFunction();
	gmac::exitGmac();
    return ret;
}
#endif

size_t
gmacAccs()
{
    size_t ret;
	gmac::enterGmac();
    gmac::trace::Function::start("GMAC", "gmacAccs");
    gmac::Process &proc = gmac::Process::current();
    ret = proc.nAccelerators();
    gmac::trace::Function::end("GMAC");
	gmac::exitGmac();
	return ret;
}

gmacError_t
gmacMigrate(int acc)
{
	gmacError_t ret = gmacSuccess;
	gmac::enterGmacExclusive();
    gmac::trace::Function::start("GMAC", "gmacMigrate");
    gmac::Process &proc = gmac::Process::current();
    if (gmac::Mode::hasCurrent()) {
        ret = proc.migrate(gmac::Mode::current(), acc);
    } else {
        if (proc.create(acc) == NULL) {
            ret = gmacErrorUnknown;
        } 
    }
    gmac::trace::Function::end("GMAC");
	gmac::exitGmac();
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
	gmac::enterGmac();
    gmac::trace::Function::start("GMAC","gmacMalloc");
    if(gmac::allocator != NULL && count < (paramPageSize / 2)) {
        *cpuPtr = gmac::allocator->alloc(count, __builtin_return_address(0));   
    }
    else {
	    count = (int(count) < getpagesize())? getpagesize(): count;
	    ret = gmac::manager->alloc(cpuPtr, count);
    }
    gmac::trace::Function::end("GMAC");
	gmac::exitGmac();
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
    gmac::enterGmac();
    gmac::trace::Function::start("GMAC", "gmacGlobalMalloc");
	count = (count < (size_t)getpagesize()) ? (size_t)getpagesize(): count;
	ret = gmac::manager->globalAlloc(cpuPtr, count, 1);
    gmac::trace::Function::end("GMAC");
    gmac::exitGmac();
    return ret;
#else
    return gmacErrorFeatureNotSupported;
#endif
}

gmacError_t
gmacFree(void *cpuPtr)
{
    gmacError_t ret = gmacSuccess;
	gmac::enterGmac();
    gmac::trace::Function::start("GMAC", "gmacFree");
    if(gmac::allocator == NULL || gmac::allocator->free(cpuPtr) == false)
        ret = gmac::manager->free(cpuPtr);
    gmac::trace::Function::end("GMAC");
	gmac::exitGmac();
	return ret;
}

void *
gmacPtr(void *ptr)
{
    void *ret = NULL;
    gmac::enterGmac();
    gmac::Process &proc = gmac::Process::current();
    ret = proc.translate(ptr);
    gmac::exitGmac();
    return ret;
}

gmacError_t
gmacLaunch(gmacKernel_t k)
{
    gmac::enterGmac();
    gmac::Mode &mode = gmac::Mode::current();
    gmac::trace::Function::start("GMAC", "gmacLaunch");
    gmac::KernelLaunch * launch = mode.launch(k);

    gmacError_t ret = gmacSuccess;
    gmac::util::Logger::TRACE("Flush the memory used in the kernel");
    gmac::util::Logger::ASSERTION(gmac::manager->release() == gmacSuccess);

    // Wait for pending transfers
    mode.sync();
    gmac::util::Logger::TRACE("Kernel Launch");
    ret = mode.execute(*launch);

    if(paramAcquireOnWrite) {
        gmac::util::Logger::TRACE("Invalidate the memory used in the kernel");
        //gmac::manager->invalidate();
    }

    delete launch;
    gmac::trace::Function::end("GMAC");
    gmac::exitGmac();

    return ret;
}

gmacError_t
gmacThreadSynchronize()
{
	gmac::enterGmac();
    gmac::trace::Function::start("GMAC", "gmacSync");

	gmacError_t ret = gmac::Mode::current().sync();
    gmac::util::Logger::TRACE("Memory Sync");
    gmac::manager->acquire();

    gmac::trace::Function::end("GMAC");
	gmac::exitGmac();
	return ret;
}

gmacError_t
gmacGetLastError()
{
	gmac::enterGmac();
	gmacError_t ret = gmac::Mode::current().error();
	gmac::exitGmac();
	return ret;
}

void *
gmacMemset(void *s, int c, size_t n)
{
    gmac::enterGmac();
    void *ret = s;
    gmac::manager->memset(s, c, n);
	gmac::exitGmac();
    return ret;
}

void *
gmacMemcpy(void *dst, const void *src, size_t n)
{
	gmac::enterGmac();
	void *ret = dst;

    gmacError_t err;

	// Locate memory regions (if any)
    gmac::Process &proc = gmac::Process::current();
    gmac::Mode *dstMode = proc.owner(dst);
    gmac::Mode *srcMode = proc.owner(src);
	if (dstMode == NULL && srcMode == NULL) return memcpy(dst, src, n);;
    gmac::manager->memcpy(dst, src, n);

	gmac::exitGmac();
	return ret;
}

void
gmacSend(pthread_t id)
{
    gmac::enterGmac();
    gmac::Process &proc = gmac::Process::current();
    proc.send((THREAD_ID)id);
    gmac::exitGmac();
}

void gmacReceive()
{
    gmac::enterGmac();
    gmac::Process &proc = gmac::Process::current();
    proc.receive();
    gmac::exitGmac();
}

void
gmacSendReceive(pthread_t id)
{
	gmac::enterGmac();
    gmac::Process &proc = gmac::Process::current();
	proc.sendReceive((THREAD_ID)id);
	gmac::exitGmac();
}

void gmacCopy(pthread_t id)
{
    gmac::enterGmac();
    gmac::Process &proc = gmac::Process::current();
    proc.copy((THREAD_ID)id);
    gmac::exitGmac();
}

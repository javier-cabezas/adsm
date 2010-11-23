#include <cstdlib>

#include "include/gmac.h"

#include "config/config.h"
#include "config/order.h"

#include "util/Logger.h"

#include "core/Context.h"
#include "core/Mode.h"
#include "core/IOBuffer.h"
#include "core/Kernel.h"
#include "core/Process.h"

#include "memory/Manager.h"
#include "memory/Allocator.h"

#include "trace/Tracer.h"

#if defined(GMAC_DLL)
#include "init.h"
#endif

#if defined(__GNUC__)
#define RETURN_ADDRESS __builtin_return_address(0)
#elif defined(_MSC_VER)
extern "C" void * _ReturnAddress(void);
#pragma intrinsic(_ReturnAddress)
#define RETURN_ADDRESS _ReturnAddress()
static long getpagesize (void) {
    static long pagesize = 0;
    if(pagesize == 0) {
        SYSTEM_INFO systemInfo;
        GetSystemInfo(&systemInfo);
        pagesize = systemInfo.dwPageSize;
    }
    return pagesize;
}
#endif

#if 0
gmacError_t
gmacClear(gmacKernel_t k)
{
    gmacError_t ret = gmacSuccess;
    gmac::enterGmac();
    enterFunction(FuncGmacClear);
    gmac::Kernel *kernel = core::Mode::current()->kernel(k);
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
    gmac::Kernel *kernel = core::Mode::current()->kernel(k);

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
    gmac::Kernel  * kernel = core::Mode::current()->kernel(k);
    if (kernel == NULL) ret = gmacErrorInvalidValue;
    else ret = kernel->unbind(obj);
	exitFunction();
	gmac::exitGmac();
    return ret;
}
#endif

size_t APICALL gmacAccs()
{
    size_t ret;
	gmac::enterGmac();
    gmac::trace::EnterCurrentFunction();
    __impl::core::Process &proc = __impl::core::Process::getInstance();
    ret = proc.nAccelerators();
    gmac::trace::ExitCurrentFunction();
	gmac::exitGmac();
	return ret;
}
#if 0
gmacError_t APICALL gmacMigrate(int acc)
{
	gmacError_t ret = gmacSuccess;
	gmac::enterGmacExclusive();
    gmac::trace::EnterCurrentFunction();
    __impl::core::Process &proc = __impl::core::Process::getInstance();
    if (__impl::core::Mode::hasCurrent()) {
        ret = proc.migrate(__impl::core::Mode::current(), acc);
    } else {
        if (proc.createMode(acc) == NULL) {
            ret = gmacErrorUnknown;
        } 
    }
    gmac::trace::ExitCurrentFunction();
	gmac::exitGmac();
	return ret;
}

gmacError_t APICALL gmacMap(void *cpuPtr, size_t count, GmacProtection prot)
{
#ifndef USE_MMAP
    gmacError_t ret = gmacSuccess;
    if (count == 0) {
        return ret;
    }
	gmac::enterGmac();
    gmac::trace::EnterCurrentFunction();
    gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();
    // TODO Remove alignment constraints
    count = (int(count) < getpagesize())? getpagesize(): count;
    ret = manager.map(cpuPtr, count, prot);
    gmac::trace::ExitCurrentFunction();
	gmac::exitGmac();
    return ret;
#else
    return gmacErrorFeatureNotSupported;
#endif
}

gmacError_t APICALL gmacUnmap(void *cpuPtr, size_t count)
{
#ifndef USE_MMAP
    gmacError_t ret = gmacSuccess;
    if (count == 0) {
        return ret;
    }
	gmac::enterGmac();
    gmac::trace::EnterCurrentFunction();
    gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();
    // TODO Remove alignment constraints
    count = (int(count) < getpagesize())? getpagesize(): count;
    ret = manager.unmap(cpuPtr, count);
    gmac::trace::ExitCurrentFunction();
	gmac::exitGmac();
    return ret;
#else
    return gmacErrorFeatureNotSupported;
#endif
}
#endif
gmacError_t APICALL gmacMalloc(void **cpuPtr, size_t count)
{
    gmacError_t ret = gmacSuccess;
    if (count == 0) {
        *cpuPtr = NULL;
        return ret;
    }
	gmac::enterGmac();
    gmac::trace::EnterCurrentFunction();
    __impl::memory::Allocator &allocator = __impl::memory::Allocator::getInstance();
    if(count < (paramPageSize / 2)) {
        *cpuPtr = allocator.alloc(count, RETURN_ADDRESS);
    }
    else {
    	gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();
	    count = (int(count) < getpagesize())? getpagesize(): count;
	    ret = manager.alloc(cpuPtr, count);
    }
    gmac::trace::ExitCurrentFunction();
	gmac::exitGmac();
	return ret;
}

gmacError_t APICALL __gmacGlobalMalloc(void **cpuPtr, size_t count, GmacGlobalMallocType hint, ...)
{
    gmacError_t ret = gmacSuccess;
    if(count == 0) {
        *cpuPtr = NULL;
        return ret;
    }
    gmac::enterGmac();
    gmac::trace::EnterCurrentFunction();
	count = (count < (size_t)getpagesize()) ? (size_t)getpagesize(): count;
	ret = gmac::memory::Manager::getInstance().globalAlloc(cpuPtr, count, hint);
    gmac::trace::ExitCurrentFunction();
    gmac::exitGmac();
    return ret;
}

gmacError_t APICALL gmacFree(void *cpuPtr)
{
    gmacError_t ret = gmacSuccess;
	gmac::enterGmac();
    gmac::trace::EnterCurrentFunction();
    __impl::memory::Allocator &allocator = __impl::memory::Allocator::getInstance();
    if (allocator.free(cpuPtr) == false) {
    	gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();
        ret = manager.free(cpuPtr);
    }
    gmac::trace::ExitCurrentFunction();
	gmac::exitGmac();
	return ret;
}

void * APICALL gmacPtr(const void *ptr)
{
    void *ret = NULL;
    gmac::enterGmac();
    ret = __impl::memory::Manager::getInstance().translate(ptr);
    gmac::exitGmac();
    return ret;
}

gmacError_t APICALL gmacLaunch(gmacKernel_t k)
{
    gmac::enterGmac();
    __impl::core::Mode &mode = __impl::core::Mode::current();
    gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();
    gmac::trace::EnterCurrentFunction();
    __impl::core::KernelLaunch &launch = mode.launch(k);

    gmacError_t ret = gmacSuccess;
    TRACE(GLOBAL, "Flush the memory used in the kernel");
    CFATAL(manager.release() == gmacSuccess, "Error releasing objects");

    // Wait for pending transfers
    mode.sync();
    TRACE(GLOBAL, "Kernel Launch");
    ret = mode.execute(launch);

    if(paramAcquireOnWrite) {
        TRACE(GLOBAL, "Invalidate the memory used in the kernel");
        //manager.invalidate();
    }

    delete &launch;
    gmac::trace::ExitCurrentFunction();
    gmac::exitGmac();

    return ret;
}

gmacError_t APICALL gmacThreadSynchronize()
{
	gmac::enterGmac();
    gmac::trace::EnterCurrentFunction();

	gmacError_t ret = __impl::core::Mode::current().sync();
    TRACE(GLOBAL, "Memory Sync");
    gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();
    manager.acquire();

    gmac::trace::ExitCurrentFunction();
	gmac::exitGmac();
	return ret;
}

gmacError_t APICALL gmacGetLastError()
{
	gmac::enterGmac();
	gmacError_t ret = __impl::core::Mode::current().error();
	gmac::exitGmac();
	return ret;
}

void * APICALL gmacMemset(void *s, int c, size_t n)
{
    gmac::enterGmac();
    void *ret = s;
    gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();
    manager.memset(s, c, n);
	gmac::exitGmac();
    return ret;
}

void * APICALL gmacMemcpy(void *dst, const void *src, size_t n)
{
	gmac::enterGmac();
	void *ret = dst;

	// Locate memory regions (if any)
    __impl::core::Process &proc = __impl::core::Process::getInstance();
    __impl::core::Mode *dstMode = proc.owner(dst);
    __impl::core::Mode *srcMode = proc.owner(src);
	if (dstMode == NULL && srcMode == NULL) return memcpy(dst, src, n);
	gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();
    manager.memcpy(dst, src, n);

	gmac::exitGmac();
	return ret;
}

void APICALL gmacSend(THREAD_T id)
{
    gmac::enterGmac();
    __impl::core::Process &proc = __impl::core::Process::getInstance();
    proc.send((THREAD_T)id);
    gmac::exitGmac();
}

void APICALL gmacReceive()
{
    gmac::enterGmac();
    __impl::core::Process &proc = __impl::core::Process::getInstance();
    proc.receive();
    gmac::exitGmac();
}

void APICALL gmacSendReceive(THREAD_T id)
{
	gmac::enterGmac();
    __impl::core::Process &proc = __impl::core::Process::getInstance();
	proc.sendReceive((THREAD_T)id);
	gmac::exitGmac();
}

void APICALL gmacCopy(THREAD_T id)
{
    gmac::enterGmac();
    __impl::core::Process &proc = __impl::core::Process::getInstance();
    proc.copy((THREAD_T)id);
    gmac::exitGmac();
}

#include <cstdlib>

#ifdef USE_CUDA
#include "include/gmac/cuda.h"
#else
#include "include/gmac/opencl.h"
#endif

#include "config/config.h"
#include "config/order.h"

#include "util/Atomics.h"
#include "util/Logger.h"

#include "core/IOBuffer.h"

#include "core/hpe/Mode.h"
#include "core/hpe/Kernel.h"
#include "core/hpe/Process.h"

#include "memory/Manager.h"
#include "memory/Allocator.h"
#ifdef DEBUG
#include "memory/protocol/common/BlockState.h"
#endif

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

using __impl::util::params::ParamBlockSize;
using __impl::util::params::ParamAutoSync;


unsigned APICALL gmacGetNumberOfAccelerators()
{
    unsigned ret;
	enterGmac();
    gmac::trace::EnterCurrentFunction();
    ret = unsigned(getProcess().nAccelerators());
    gmac::trace::ExitCurrentFunction();
	exitGmac();
	return ret;
}

size_t APICALL gmacGetFreeMemory()
{
    enterGmacExclusive();
    gmac::trace::EnterCurrentFunction();
    size_t free;
    size_t total;
    getCurrentMode().memInfo(free, total);
    gmac::trace::ExitCurrentFunction();
    exitGmac();
    return free;
}


gmacError_t APICALL gmacMigrate(unsigned acc)
{
	gmacError_t ret = gmacSuccess;
	enterGmacExclusive();
    gmac::trace::EnterCurrentFunction();
    ret = getProcess().migrate(acc);
    gmac::trace::ExitCurrentFunction();
	exitGmac();
	return ret;
}


gmacError_t APICALL gmacMemoryMap(void *cpuPtr, size_t count, GmacProtection prot)
{
#if 0
    gmacError_t ret = gmacSuccess;
    if (count == 0) {
        return ret;
    }
	enterGmac();
    gmac::trace::EnterCurrentFunction();
    // TODO Remove alignment constraints
    count = (int(count) < getpagesize())? getpagesize(): count;
    ret = getManager().map(cpuPtr, count, prot);
    gmac::trace::ExitCurrentFunction();
	exitGmac();
    return ret;
#endif
    return gmacErrorFeatureNotSupported;
}


gmacError_t APICALL gmacMemoryUnmap(void *cpuPtr, size_t count)
{
#if 0
    gmacError_t ret = gmacSuccess;
    if (count == 0) {
        return ret;
    }
	enterGmac();
    gmac::trace::EnterCurrentFunction();
    // TODO Remove alignment constraints
    count = (int(count) < getpagesize())? getpagesize(): count;
    ret = getManager().unmap(cpuPtr, count);
    gmac::trace::ExitCurrentFunction();
	exitGmac();
    return ret;
#endif
    return gmacErrorFeatureNotSupported;
}


gmacError_t APICALL gmacMalloc(void **cpuPtr, size_t count)
{
    gmacError_t ret = gmacSuccess;
    if (count == 0) {
        *cpuPtr = NULL;
        return ret;
    }
	enterGmac();
    gmac::trace::EnterCurrentFunction();
    if(count < (ParamBlockSize / 2)) {
        *cpuPtr = getAllocator().alloc(getCurrentMode(), count, hostptr_t(RETURN_ADDRESS));
    }
    else {
	    count = (int(count) < getpagesize())? getpagesize(): count;
	    ret = getManager().alloc(getCurrentMode(), (hostptr_t *) cpuPtr, count);
    }
    gmac::trace::ExitCurrentFunction();
	exitGmac();
	return ret;
}

gmacError_t APICALL gmacGlobalMalloc(void **cpuPtr, size_t count, GmacGlobalMallocType hint)
{
    gmacError_t ret = gmacSuccess;
    if(count == 0) {
        *cpuPtr = NULL;
        return ret;
    }
    enterGmac();
    gmac::trace::EnterCurrentFunction();
	count = (count < (size_t)getpagesize()) ? (size_t)getpagesize(): count;
	ret = getManager().globalAlloc(getCurrentMode(), (hostptr_t *)cpuPtr, count, hint);
    gmac::trace::ExitCurrentFunction();
    exitGmac();
    return ret;
}

gmacError_t APICALL gmacFree(void *cpuPtr)
{
    gmacError_t ret = gmacSuccess;
	enterGmac();
    gmac::trace::EnterCurrentFunction();
    __impl::core::hpe::Mode &mode = getCurrentMode();
    if(getAllocator().free(mode, hostptr_t(cpuPtr)) == false) {
        ret = getManager().free(mode, hostptr_t(cpuPtr));
    }
    gmac::trace::ExitCurrentFunction();
	exitGmac();
	return ret;
}

__gmac_accptr_t APICALL gmacPtr(const void *ptr)
{
    accptr_t ret = accptr_t(0);
    enterGmac();
    ret = getManager().translate(getCurrentMode(), hostptr_t(ptr));
    exitGmac();
    TRACE(GLOBAL, "Translate %p to %p", ptr, ret.get());
    return ret.get();
}

gmacError_t GMAC_LOCAL gmacLaunch(__impl::core::hpe::KernelLaunch &launch)
{
    gmacError_t ret = gmacSuccess;
    __impl::core::hpe::Mode &mode = launch.getMode();
    Manager &manager = getManager();
    TRACE(GLOBAL, "Flush the memory used in the kernel");
    ret = manager.releaseObjects(mode);
    CFATAL(ret == gmacSuccess, "Error releasing objects");

    TRACE(GLOBAL, "Kernel Launch");
    ret = mode.execute(launch);

    if (ParamAutoSync == true) {
        TRACE(GLOBAL, "Waiting for Kernel to complete");
        mode.wait();
        TRACE(GLOBAL, "Memory Sync");
        ret = manager.acquireObjects(getCurrentMode());
        CFATAL(ret == gmacSuccess, "Error waiting for kernel");
    }

    return ret;
}

gmacError_t APICALL gmacLaunch(gmac_kernel_id_t k)
{
    enterGmac();
    gmac::trace::EnterCurrentFunction();
    __impl::core::hpe::Mode &mode = getCurrentMode();
    __impl::core::hpe::KernelLaunch *launch = NULL;
    gmacError_t ret = mode.launch(k, launch);

    if(ret == gmacSuccess) {
        ret = gmacLaunch(*launch);
        delete launch;
    }

    gmac::trace::ExitCurrentFunction();
    exitGmac();

    return ret;
}

gmacError_t GMAC_LOCAL gmacThreadSynchronize(__impl::core::hpe::KernelLaunch &launch)
{
    gmacError_t ret = gmacSuccess;
    if(ParamAutoSync == false) {
        __impl::core::hpe::Mode &mode = getCurrentMode();
        mode.wait(launch);
        TRACE(GLOBAL, "Memory Sync");
        ret = getManager().acquireObjects(mode);
    }
    return ret;
}

gmacError_t APICALL gmacThreadSynchronize()
{
	enterGmac();
    gmac::trace::EnterCurrentFunction();

	gmacError_t ret = gmacSuccess;
    if (ParamAutoSync == false) {
        __impl::core::hpe::Mode &mode = getCurrentMode();
        mode.wait();
        TRACE(GLOBAL, "Memory Sync");
        ret = getManager().acquireObjects(mode);
    }

    gmac::trace::ExitCurrentFunction();
	exitGmac();
	return ret;
}

gmacError_t APICALL gmacGetLastError()
{
	enterGmac();
	gmacError_t ret = getCurrentMode().error();
	exitGmac();
	return ret;
}

void * APICALL gmacMemset(void *s, int c, size_t size)
{
    enterGmac();
    void *ret = s;
    getManager().memset(getCurrentMode(), hostptr_t(s), c, size);
	exitGmac();
    return ret;
}

void * APICALL gmacMemcpy(void *dst, const void *src, size_t size)
{
	enterGmac();
	void *ret = dst;

	// Locate memory regions (if any)
    Process &proc = getProcess();
    __impl::core::Mode *dstMode = proc.owner(hostptr_t(dst), size);
    __impl::core::Mode *srcMode = proc.owner(hostptr_t(src), size);
    if (dstMode == NULL && srcMode == NULL) {
        exitGmac();
        return ::memcpy(dst, src, size);
    }
    getManager().memcpy(getCurrentMode(), hostptr_t(dst), hostptr_t(src), size);

	exitGmac();
	return ret;
}

void APICALL gmacSend(THREAD_T id)
{
    enterGmac();
    getProcess().send((THREAD_T)id);
    exitGmac();
}

void APICALL gmacReceive()
{
    enterGmac();
    getProcess().receive();
    exitGmac();
}

void APICALL gmacSendReceive(THREAD_T id)
{
	enterGmac();
	getProcess().sendReceive((THREAD_T)id);
	exitGmac();
}

void APICALL gmacCopy(THREAD_T id)
{
    enterGmac();
    getProcess().copy((THREAD_T)id);
    exitGmac();
}

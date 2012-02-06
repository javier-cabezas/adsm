/**
 * \file src/hpe/gmac.cpp
 *
 * Implementation of the generic HPE API calls
 */

#include <cstdlib>

#ifdef USE_CUDA
#include "include/gmac/cuda.h"
#else
#include "include/gmac/opencl.h"
#endif

#include "config/config.h"
#include "config/order.h"

#include "hal/types.h"

#include "hpe/core/address_space.h"
#include "hpe/core/kernel.h"
#include "hpe/core/process.h"
#include "hpe/core/thread.h"
#include "hpe/core/vdevice.h"

#include "memory/allocator.h"
#include "memory/manager.h"
#ifdef DEBUG
#include "memory/protocols/common/block.h"
#endif

#include "trace/logger.h"
#include "trace/Tracer.h"

#include "util/atomics.h"

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

using namespace __impl::core::hpe;
using namespace __impl::memory;
using namespace __impl::util;

using config::params::BlockSize;
using config::params::AutoSync;

static inline
__impl::core::hpe::resource_manager &
get_resource_manager()
{
    return get_process().get_resource_manager();
}

GMAC_API unsigned APICALL
gmacGetNumberOfDevices()
{
    unsigned ret;
    enterGmac();
    gmac::trace::EnterCurrentFunction();
    ret = unsigned(get_resource_manager().get_number_of_devices());
    gmac::trace::ExitCurrentFunction();
    exitGmac();
    return ret;
}

GMAC_API gmacError_t APICALL
gmacGetDeviceInfo(unsigned acc, GmacDeviceInfo *info)
{
    enterGmacExclusive();
    gmac::trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;
    __impl::core::hpe::resource_manager &resourceManager = get_resource_manager();
    if (acc < resourceManager.get_number_of_devices() && info != NULL) {
    	ret = get_resource_manager().get_device_info(acc, *info);
    } else {
        ret = gmacErrorInvalidValue;
    }
    gmac::trace::ExitCurrentFunction();
    thread::get_current_thread().set_last_error(ret);
    exitGmac();
    return ret;
}

GMAC_API unsigned APICALL
gmacGetCurrentDeviceId()
{
    unsigned ret;
    enterGmac();
    gmac::trace::EnterCurrentFunction();
    ret = thread::get_current_thread().get_current_virtual_device().get_address_space()->get_hal_context().get_device().get_print_id();
    gmac::trace::ExitCurrentFunction();
    exitGmac();
    return ret;
}

GMAC_API gmacError_t APICALL
gmacGetFreeMemory(unsigned acc, size_t *freeMemory)
{
    enterGmacExclusive();
    gmac::trace::EnterCurrentFunction();
    gmacError_t ret = get_resource_manager().get_device_free_mem(acc, *freeMemory);
    gmac::trace::ExitCurrentFunction();
    thread::get_current_thread().set_last_error(ret);
    exitGmac();
    return ret;
}

#if 0
GMAC_API gmacError_t APICALL
gmacMigrate(unsigned acc)
{
    gmacError_t ret = gmacSuccess;
    enterGmacExclusive();
    gmac::trace::EnterCurrentFunction();
    ret = get_process().migrate(acc);
    gmac::trace::ExitCurrentFunction();
    thread::get_current_thread().set_last_error(ret);
    exitGmac();
    return ret;
}
#endif

extern "C"
GMAC_API gmacError_t APICALL
gmacCreateAddressSpace(GmacAddressSpaceId *aspaceId, int accId)
{
    gmacError_t ret = gmacSuccess;
    enterGmac();
    gmac::trace::EnterCurrentFunction();
    if ((accId == ADDRESS_SPACE_ACCELERATOR_ANY) ||
        (accId >= 0 && accId < int(get_resource_manager().get_number_of_devices()))) {
        __impl::core::hpe::address_space_ptr aspace = get_resource_manager().create_address_space(accId, ret);
        if (ret == gmacSuccess) {
            ASSERTION(bool(aspace));
            *aspaceId = aspace->get_id();
        }
    } else {
        ret = gmacErrorInvalidValue;
    }
    gmac::trace::ExitCurrentFunction();
    exitGmac();

    return ret;
}

extern "C"
GMAC_API gmacError_t APICALL
gmacDeleteAddressSpace(GmacAddressSpaceId aspaceId)
{
    gmacError_t ret = gmacSuccess;
    enterGmac();
    gmac::trace::EnterCurrentFunction();

    __impl::core::hpe::address_space_ptr aspace = get_resource_manager().get_address_space(aspaceId);
    if (aspace) {
        ret = get_resource_manager().destroy_address_space(*aspace);
    } else {
        ret = gmacErrorInvalidValue;
    }
    gmac::trace::ExitCurrentFunction();
    exitGmac();

    return ret;
}

extern "C"
GMAC_API gmacError_t APICALL
gmacCreateVirtualDevice(GmacVirtualDeviceId *vDeviceId, GmacAddressSpaceId aspaceId)
{
    gmacError_t ret = gmacSuccess;
    enterGmac();
    gmac::trace::EnterCurrentFunction();
    vdevice *dev = get_resource_manager().create_virtual_device(aspaceId, ret);
    if (ret == gmacSuccess) {
        *vDeviceId = dev->get_id();
    }
    gmac::trace::ExitCurrentFunction();
    exitGmac();

    return ret;
}

extern "C"
GMAC_API gmacError_t APICALL
gmacDeleteVirtualDevice(GmacVirtualDeviceId vDeviceId)
{
    gmacError_t ret = gmacSuccess;
    enterGmac();
    gmac::trace::EnterCurrentFunction();
    vdevice *dev = thread::get_current_thread().get_virtual_device(vDeviceId);
    if (dev != NULL) {
        ret = get_resource_manager().destroy_virtual_device(*dev);
    } else {
        ret = gmacErrorInvalidValue;
    }
    gmac::trace::ExitCurrentFunction();
    exitGmac();

    return ret;
}

extern "C"
GMAC_API gmacError_t APICALL
gmacSetVirtualDevice(GmacVirtualDeviceId vDeviceId)
{
    gmacError_t ret = gmacSuccess;
    enterGmac();
    gmac::trace::EnterCurrentFunction();
    vdevice *dev = thread::get_current_thread().get_virtual_device(vDeviceId);
    if (dev != NULL) {
        thread::get_current_thread().set_current_virtual_device(*dev);
    } else {
        ret = gmacErrorInvalidValue;
    }
    gmac::trace::ExitCurrentFunction();
    exitGmac();

    return ret;
}

extern "C"
GMAC_API gmacError_t APICALL
gmacMemoryMap(void *cpuPtr, size_t count, GmacProtection prot)
{
#if 0
    gmacError_t ret = gmacSuccess;
    if (count == 0) {
        return ret;
    }
        enterGmac();
    gmac::trace::EnterCurrentFunction();
    // TODO Remove alignment constraints?
    count = (int(count) < getpagesize())? getpagesize(): count;
    ret = get_manager().map(cpuPtr, count, prot);
    gmac::trace::ExitCurrentFunction();
        exitGmac();
    return ret;
#endif
    gmacError_t ret = gmacErrorFeatureNotSupported;
    thread::get_current_thread().set_last_error(ret);
    return ret;
}

extern "C"
GMAC_API gmacError_t APICALL
gmacMemoryUnmap(void *cpuPtr, size_t count)
{
#if 0
    gmacError_t ret = gmacSuccess;
    if (count == 0) {
        return ret;
    }
        enterGmac();
    gmac::trace::EnterCurrentFunction();
    // TODO Remove alignment constraints?
    count = (int(count) < getpagesize())? getpagesize(): count;
    ret = get_manager().unmap(cpuPtr, count);
    gmac::trace::ExitCurrentFunction();
        exitGmac();
    return ret;
#endif
    gmacError_t ret = gmacErrorFeatureNotSupported;
    thread::get_current_thread().set_last_error(ret);
    return ret;
}

extern "C"
GMAC_API gmacError_t APICALL
gmacMalloc(void **cpuPtr, size_t count)
{
    gmacError_t ret = gmacSuccess;
    if (count == 0) {
        *cpuPtr = NULL;
        thread::get_current_thread().set_last_error(ret);
        return ret;
    }
    enterGmac();
    gmac::trace::EnterCurrentFunction();
    if(has_allocator() && count < (BlockSize / 2)) {
        *cpuPtr = get_allocator().alloc(thread::get_current_thread().get_current_virtual_device().get_address_space(), count, host_ptr(RETURN_ADDRESS));
    }
    else {
        count = (int(count) < getpagesize())? getpagesize(): count;
        ret = get_manager().alloc(thread::get_current_thread().get_current_virtual_device().get_address_space(), (host_ptr *) cpuPtr, count);
    }
    gmac::trace::ExitCurrentFunction();
    thread::get_current_thread().set_last_error(ret);
    exitGmac();
    return ret;
}

#if 0
GMAC_API gmacError_t APICALL
gmacGlobalMalloc(void **cpuPtr, size_t count, GmacGlobalMallocType hint)
{
    gmacError_t ret = gmacSuccess;
    if(count == 0) {
        *cpuPtr = NULL;
        thread::get_current_thread().set_last_error(ret);
        return ret;
    }
    enterGmac();
    gmac::trace::EnterCurrentFunction();
    count = (count < (size_t)getpagesize()) ? (size_t)getpagesize(): count;
    ret = get_manager().globalAlloc(thread::get_current_thread().get_current_virtual_device().get_address_space(), (host_ptr *)cpuPtr, count, hint);
    gmac::trace::ExitCurrentFunction();
    thread::get_current_thread().set_last_error(ret);
    exitGmac();
    return ret;
}
#endif

extern "C"
GMAC_API gmacError_t APICALL
gmacFree(void *cpuPtr)
{
    gmacError_t ret = gmacSuccess;
    enterGmac();
    if(cpuPtr == NULL) {
        thread::get_current_thread().set_last_error(ret);
        exitGmac();
        return ret;
    }
    gmac::trace::EnterCurrentFunction();
    __impl::core::hpe::address_space_ptr aspace = thread::get_current_thread().get_current_virtual_device().get_address_space();
    if(has_allocator() == false || get_allocator().free(aspace, host_ptr(cpuPtr)) == false) {
        ret = get_manager().free(aspace, host_ptr(cpuPtr));
    }
    gmac::trace::ExitCurrentFunction();
    thread::get_current_thread().set_last_error(ret);
    exitGmac();
    return ret;
}

#ifdef USE_CUDA
extern "C"
GMAC_API __gmac_accptr_t APICALL
gmacPtr(const void *ptr)
{
    __impl::hal::ptr ret;
    enterGmac();
    ret = get_manager().translate(thread::get_current_thread().get_current_virtual_device().get_address_space(), host_ptr(ptr));
    exitGmac();
    TRACE(GLOBAL, "Translate %p to %p", ptr, ret.get_base() + ret.get_offset());
    return __gmac_accptr_t(ret.get_base() + ret.get_offset());
}
#endif

gmacError_t GMAC_LOCAL
gmacLaunch(__impl::core::hpe::kernel::launch_ptr launch)
{
    gmacError_t ret = gmacSuccess;
    vdevice &dev = launch->get_virtual_device();
    __impl::core::hpe::address_space_ptr aspace = dev.get_address_space();
    manager &manager = get_manager();
    TRACE(GLOBAL, "Flush the memory used in the kernel");
    const std::list<__impl::memory::object_access_info> &objects = launch->get_arg_list().get_objects();
    // If the launch object does not contain objects, assume all the objects
    // in the mode are released
    ret = manager.release_objects(aspace, objects);
    CFATAL(ret == gmacSuccess, "Error releasing objects");

    TRACE(GLOBAL, "Kernel Launch");
    /* __impl::hal::async_event_t *event = */
        dev.execute(launch, ret);

    if (AutoSync == true) {
        TRACE(GLOBAL, "Waiting for Kernel to complete");
        // TODO: wait for the event instead for the device
        dev.wait();
        TRACE(GLOBAL, "Memory Sync");
        ret = manager.acquire_objects(aspace, objects);
        CFATAL(ret == gmacSuccess, "Error waiting for kernel");
    }

    thread::get_current_thread().set_last_error(ret);

    return ret;
}

extern "C"
GMAC_API gmacError_t APICALL
gmacLaunch(gmac_kernel_id_t k, __impl::hal::kernel_t::config &config, __impl::core::hpe::kernel::arg_list &args)
{
    enterGmac();
    gmac::trace::EnterCurrentFunction();
    vdevice &dev = thread::get_current_thread().get_current_virtual_device();
    kernel::launch_ptr launch;
    gmacError_t ret;
    launch = dev.launch(k, config, args, ret);

    if(ret == gmacSuccess) {
        ret = gmacLaunch(launch);
    }

    gmac::trace::ExitCurrentFunction();

    thread::get_current_thread().set_last_error(ret);
    exitGmac();

    return ret;
}

gmacError_t GMAC_LOCAL
gmacThreadSynchronize(kernel::launch_ptr launch)
{
    gmacError_t ret = gmacSuccess;
    if(AutoSync == false) {
        vdevice &dev = thread::get_current_thread().get_current_virtual_device();
        dev.wait(launch);
        TRACE(GLOBAL, "Memory Sync");
        ret = get_manager().acquire_objects(dev.get_address_space(), launch->get_arg_list().get_objects());
    }
    return ret;
}

extern "C"
GMAC_API gmacError_t APICALL
gmacThreadSynchronize()
{
    enterGmac();
    gmac::trace::EnterCurrentFunction();

    gmacError_t ret = gmacSuccess;
    if (AutoSync == false) {
        vdevice &dev = thread::get_current_thread().get_current_virtual_device();
        __impl::core::hpe::address_space_ptr aspace = dev.get_address_space();
        dev.wait();
        TRACE(GLOBAL, "Memory Sync");
        ret = get_manager().acquire_objects(aspace);
    }

    gmac::trace::ExitCurrentFunction();
    thread::get_current_thread().set_last_error(ret);
    exitGmac();
    return ret;
}

extern "C"
GMAC_API gmacError_t APICALL
gmacGetLastError()
{
    enterGmac();
    gmacError_t ret = thread::get_current_thread().get_last_error();
    exitGmac();
    return ret;
}

extern "C"
/** \todo Move to a more CUDA-like API */
GMAC_API void * APICALL
gmacMemset(void *s, int c, size_t size)
{
    enterGmac();
    void *ret = s;
    get_manager().memset(thread::get_current_thread().get_current_virtual_device().get_address_space(), host_ptr(s), c, size);
    exitGmac();
    return ret;
}

extern "C"
/** \todo Move to a more CUDA-like API */
GMAC_API void * APICALL
gmacMemcpy(void *dst, const void *src, size_t size)
{
    enterGmac();
    void *ret = dst;

    // Locate memory regions (if any)
    __impl::memory::address_space_ptr aspaceDst = get_manager().get_owner(host_ptr(dst), size);
    __impl::memory::address_space_ptr aspaceSrc = get_manager().get_owner(host_ptr(src), size);
    if (!aspaceDst && !aspaceSrc) {
        exitGmac();
        return ::memcpy(dst, src, size);
    }
    get_manager().memcpy(thread::get_current_thread().get_current_virtual_device().get_address_space(), host_ptr(dst), host_ptr(src), size);

    exitGmac();
    return ret;
}

#if 0
/** \todo Return error */
GMAC_API void APICALL
gmacSend(THREAD_T id)
{
    enterGmac();
    get_process().send((THREAD_T)id);
    exitGmac();
}

/** \todo Return error */
GMAC_API void APICALL
gmacReceive()
{
    enterGmac();
    get_process().receive();
    exitGmac();
}

/** \todo Return error */
GMAC_API void APICALL
gmacSendReceive(THREAD_T id)
{
    enterGmac();
    get_process().sendReceive((THREAD_T)id);
    exitGmac();
}

/** \todo Return error */
GMAC_API void APICALL
gmacCopy(THREAD_T id)
{
    enterGmac();
    get_process().copy((THREAD_T)id);
    exitGmac();
}
#endif

#ifdef USE_INTERNAL_API

extern "C"
GMAC_API gmacError_t APICALL
__gmacFlushDirty()
{
    enterGmac();
    gmacError_t ret = get_manager().flush_dirty(thread::get_current_thread().get_current_virtual_device());
    thread::get_current_thread().set_last_error(ret);
    exitGmac();
    return ret;
}

#endif

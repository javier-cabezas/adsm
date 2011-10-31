#include "config/common.h"

#include "coherence_domain.h"
#include "device.h"
#include "module.h"

#define __GMAC_ERROR(r, err) case r: error = err; break

static bool initialized = false;

namespace __impl { namespace hal {

gmacError_t
init_platform()
{
    gmacError_t ret = gmacSuccess;

    if (initialized == false) {
        TRACE(GLOBAL, "Initializing CUDA Driver API");
        if (cuInit(0) != CUDA_SUCCESS) {
            FATAL("Unable to init CUDA");
        }
    }

    return ret;
}

cuda::vector_module Modules_;

std::list<cuda::device *>
init_devices()
{
    static bool initialized = false;

    if (initialized == false) {
        initialized = true;
    } else {
        FATAL("Double HAL initialization");
    }

    std::list<cuda::device *> devices;

    int devCount = 0;
    int devRealCount = 0;
    
    CUresult err = cuDeviceGetCount(&devCount);
    if(err != CUDA_SUCCESS)
        FATAL("Error getting CUDA-enabled devices");

    TRACE(GLOBAL, "Found %d CUDA capable devices", devCount);

    // Add accelerators to the system
    for(int i = 0; i < devCount; i++) {
        CUdevice cuDev;
        if(cuDeviceGet(&cuDev, i) != CUDA_SUCCESS)
            FATAL("Unable to access CUDA device");
#if CUDA_VERSION >= 2020
        int attr = 0;
        if(cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cuDev) != CUDA_SUCCESS)
            FATAL("Unable to access CUDA device");
        if(attr != CU_COMPUTEMODE_PROHIBITED) {
            cuda::coherence_domain *coherenceDomain = new cuda::coherence_domain();
            // TODO: detect coherence domain correctly. Now using one per device.
            cuda::device *dev = new cuda::device(cuDev, *coherenceDomain);
            devices.push_back(dev);
            devRealCount++;
        }
#else
        cuda::device *dev = new cuda::device(cuDev, *coherenceDomain);
        devices.push_back(dev);
        devRealCount++;
#endif
    }

    if(devRealCount == 0)
        MESSAGE("No CUDA-enabled devices found");

    return devices;
}

}}

namespace __impl { namespace hal { namespace cuda { 

gmacError_t error(CUresult err)
{
    gmacError_t error = gmacSuccess;
    switch(err) {
        __GMAC_ERROR(CUDA_SUCCESS, gmacSuccess);
        __GMAC_ERROR(CUDA_ERROR_INVALID_VALUE, gmacErrorInvalidValue);
        __GMAC_ERROR(CUDA_ERROR_OUT_OF_MEMORY, gmacErrorMemoryAllocation);
        __GMAC_ERROR(CUDA_ERROR_NOT_INITIALIZED, gmacErrorNotReady);
        __GMAC_ERROR(CUDA_ERROR_DEINITIALIZED, gmacErrorNotReady);
        __GMAC_ERROR(CUDA_ERROR_NO_DEVICE, gmacErrorNoAccelerator);
        __GMAC_ERROR(CUDA_ERROR_INVALID_DEVICE, gmacErrorInvalidAccelerator);
        __GMAC_ERROR(CUDA_ERROR_INVALID_IMAGE, gmacErrorInvalidAcceleratorFunction);
        __GMAC_ERROR(CUDA_ERROR_INVALID_CONTEXT, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_CONTEXT_ALREADY_CURRENT, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_ALREADY_MAPPED, gmacErrorMemoryAllocation);
        __GMAC_ERROR(CUDA_ERROR_NO_BINARY_FOR_GPU, gmacErrorInvalidAcceleratorFunction);
        __GMAC_ERROR(CUDA_ERROR_ALREADY_ACQUIRED, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_FILE_NOT_FOUND, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_INVALID_HANDLE, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_NOT_FOUND, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_NOT_READY, gmacErrorNotReady);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_FAILED, gmacErrorLaunchFailure);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, gmacErrorLaunchFailure);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_TIMEOUT, gmacErrorLaunchFailure);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, gmacErrorLaunchFailure);
        __GMAC_ERROR(CUDA_ERROR_UNKNOWN, gmacErrorUnknown);
        default: error = gmacErrorUnknown;
    }
    return error;
}

context_t::context_t(CUcontext ctx, device &dev) :
    Parent(ctx, dev)
{
}

accptr_t
context_t::alloc(size_t count, gmacError_t &err)
{
    set();

    CUdeviceptr devPtr = 0;
    CUresult res = cuMemAlloc(&devPtr, count);

    err = cuda::error(res);

    return accptr_t(devPtr);
}

buffer_t *
context_t::alloc_buffer(size_t count, GmacProtection hint, gmacError_t &err)
{
    set();

    // TODO: add a parater to specify accesibility of the buffer from the device
    unsigned flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;
    if (hint == GMAC_PROT_WRITE) {
        flags += CU_MEMHOSTALLOC_WRITECOMBINED;
    }
    void *addr;
    CUresult res = cuMemHostAlloc(&addr, count, flags);
    err = cuda::error(res);

    return new buffer_t(hostptr_t(addr), *this);
}

gmacError_t
context_t::free(accptr_t acc)
{
    set();

    CUresult ret = cuMemFree(acc.get());

    return cuda::error(ret);
}

gmacError_t
context_t::free_buffer(buffer_t &buffer)
{
    set();

    CUresult ret = cuMemFreeHost(buffer.get_addr());

    return cuda::error(ret);
}

event_t &
context_t::copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, list_event &dependencies)
{
    set();

    CUresult res;

    TRACE(LOCAL, "H (%p) -> D copy ("FMT_SIZE" bytes) on stream: %p", src, count, stream());
    event_t *ret = new event_t(event_t::Transfer, *this);

    ret->begin(stream);
    res = cuMemcpyHtoD(dst.get(), src, count);
    ret->end();

    if (res != CUDA_SUCCESS) {
        ret->set_error(error(res));
    }

    return *ret;
}

event_t &
context_t::copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, async_event_t &event)
{
    event_t *ret;
    gmacError_t err = event.sync();

    if (err == gmacSuccess) {
        ret = &copy(dst, src, count, stream);
    } else {
        ret = new event_t(event_t::Transfer, *this);
        ret->set_error(err);
    }

    return *ret;
}

event_t &
context_t::copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event &dependencies)
{
    set();

    CUresult res;

    TRACE(LOCAL, "D -> H (%p) copy ("FMT_SIZE" bytes) on stream: %p", dst, count, stream());
    event_t *ret = new event_t(event_t::Transfer, *this);

    ret->begin(stream);
    res = cuMemcpyDtoH(dst, src.get(), count);
    ret->end();

    if (res != CUDA_SUCCESS) {
        ret->set_error(error(res));
    }

    return *ret;
}

event_t &
context_t::copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, async_event_t &event)
{
    event_t *ret;
    gmacError_t err = event.sync();

    if (err == gmacSuccess) {
        ret = &copy(dst, src, count, stream);
    } else {
        ret = new event_t(event_t::Transfer, *this);
        ret->set_error(err);
    }

    return *ret;
}

event_t &
context_t::copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event &dependencies)
{
    set();

    CUresult res;

    TRACE(LOCAL, "D -> D copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t *ret = new event_t(event_t::Transfer, *this);

    ret->begin(stream);
    res = cuMemcpyDtoD(dst.get(), src.get(), count);
    ret->end();

    if (res != CUDA_SUCCESS) {
        ret->set_error(error(res));
    }

    return *ret;
}

event_t &
context_t::copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, async_event_t &event)
{
    event_t *ret;
    gmacError_t err = event.sync();

    if (err == gmacSuccess) {
        ret = &copy(dst, src, count, stream);
    } else {
        ret = new event_t(event_t::Transfer, *this);
        ret->set_error(err);
    }

    return *ret;
}

async_event_t &
context_t::copy_async(accptr_t dst, buffer_t buffer, size_t off, size_t count, stream_t &stream, list_event &dependencies)
{
    set();

    CUresult res;

    TRACE(LOCAL, "H (%p + "FMT_SIZE") -> D async copy ("FMT_SIZE" bytes) on stream: %p", buffer.get_addr(), off, count, stream());
    async_event_t *ret = new async_event_t(event_t::Transfer, *this);

    ret->begin(stream);
    res = cuMemcpyHtoDAsync(dst.get(), buffer.get_addr() + off, count, stream());
    ret->end();

    if (res == CUDA_SUCCESS) {
        stream.set_last_async_event(ret);
    } else {
        ret->set_error(error(res));
    }

    return *ret;
}

async_event_t &
context_t::copy_async(accptr_t dst, buffer_t buffer, size_t off, size_t count, stream_t &stream, async_event_t &event)
{
    async_event_t *ret;
    gmacError_t err = event.sync();

    if (err == gmacSuccess) {
        ret = &copy_async(dst, buffer, off, count, stream);
    } else {
        ret = new async_event_t(event_t::Transfer, *this);
        ret->set_error(err);
    }

    return *ret;
}

async_event_t &
context_t::copy_async(buffer_t buffer, size_t off, accptr_t src, size_t count, stream_t &stream, list_event &dependencies)
{
    set();

    CUresult res;

    TRACE(LOCAL, "D -> H (%p + "FMT_SIZE") async copy ("FMT_SIZE" bytes) on stream: %p", buffer.get_addr(), off, count, stream());
    async_event_t *ret = new async_event_t(event_t::Transfer, *this);

    ret->begin(stream);
    res = cuMemcpyDtoHAsync(buffer.get_addr() + off, src.get(), count, stream());
    ret->end();

    if (res == CUDA_SUCCESS) {
        stream.set_last_async_event(ret);
    } else {
        ret->set_error(error(res));
    }

    return *ret;
}

async_event_t &
context_t::copy_async(buffer_t buffer, size_t off, accptr_t src, size_t count, stream_t &stream, async_event_t &event)
{
    async_event_t *ret;
    gmacError_t err = event.sync();

    if (err == gmacSuccess) {
        ret = &copy_async(buffer, off, src, count, stream);
    } else {
        ret = new async_event_t(event_t::Transfer, *this);
        ret->set_error(err);
    }

    return *ret;
}

async_event_t &
context_t::copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event &dependencies)
{
    set();

    CUresult res;

    TRACE(LOCAL, "D -> D async copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    async_event_t *ret = new async_event_t(event_t::Transfer, *this);

    ret->begin(stream);
    res = cuMemcpyDtoDAsync(dst.get(), src.get(), count, stream());
    ret->end();

    if (res == CUDA_SUCCESS) {
        stream.set_last_async_event(ret);
    } else {
        ret->set_error(error(res));
    }

    return *ret;
}

async_event_t &
context_t::copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, async_event_t &event)
{
    async_event_t *ret;
    gmacError_t err = event.sync();

    if (err == gmacSuccess) {
        ret = &copy_async(dst, src, count, stream);
    } else {
        ret = new async_event_t(event_t::Transfer, *this);
        ret->set_error(err);
    }

    return *ret;
}

event_t &
context_t::memset(accptr_t dst, int c, size_t count, stream_t &stream, list_event &dependencies)
{
    set();

    TRACE(LOCAL, "memset ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t *ret = new event_t(event_t::Transfer, *this);

    ret->begin(stream);
    CUresult res = cuMemsetD8(dst.get(), (unsigned char)c, count);
    ret->end();

    if (res != CUDA_SUCCESS) {
        ret->set_error(error(res));
    }

    return *ret;
}

event_t &
context_t::memset(accptr_t dst, int c, size_t count, stream_t &stream, async_event_t &event)
{
    event_t *ret;
    gmacError_t err = event.sync();

    if (err == gmacSuccess) {
        ret = &memset(dst, c, count, stream);
    } else {
        ret = new event_t(event_t::Transfer, *this);
        ret->set_error(err);
    }

    return *ret;
}

async_event_t &
context_t::memset_async(accptr_t dst, int c, size_t count, stream_t &stream, list_event &dependencies)
{
    set();

    TRACE(LOCAL, "async memset ("FMT_SIZE" bytes) on stream: %p", count, stream());
    async_event_t *ret = new async_event_t(event_t::Transfer, *this);

    ret->begin(stream);
    CUresult res = cuMemsetD8(dst.get(), (unsigned char)c, count);
    ret->end();

    if (res == CUDA_SUCCESS) {
        stream.set_last_async_event(ret);
    } else {
        ret->set_error(error(res));
    }

    return *ret;
}

async_event_t &
context_t::memset_async(accptr_t dst, int c, size_t count, stream_t &stream, async_event_t &event)
{
    async_event_t *ret;
    gmacError_t err = event.sync();

    if (err == gmacSuccess) {
        ret = &memset_async(dst, c, count, stream);
    } else {
        ret = new async_event_t(event_t::Transfer, *this);
        ret->set_error(err);
    }

    return *ret;
}

const code_repository &
context_t::get_code_repository() const
{
    if (Modules_.size() == 0) {
        Modules_ = module_descriptor::create_modules();
    }
    ASSERTION(Modules_.size() == 1, "More than one module found!");

    return *Modules_[0];
}

stream_t::stream_t(CUstream stream, context_t &context) :
    Parent(stream, context)
{
}

gmacError_t
stream_t::sync()
{
    get_context().set(); 

    TRACE(LOCAL, "Waiting for stream: %p", (*this)());
    CUresult ret = cuStreamSynchronize((*this)());

    return cuda::error(ret);
}

async_event_t::Parent::state
async_event_t::get_state()
{
    if (Parent::state_ != Parent::End) {
        get_stream().get_context().set();

        CUresult res = cuEventQuery(eventEnd_);

        if (res == CUDA_ERROR_NOT_READY) {
            state_ = Queued;
        } else if (res == CUDA_SUCCESS) {
            state_ = End;
        }
    }

    return state_;
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

#include "util/Logger.h"

#include "hal/types.h"
#include "device.h"

namespace __impl { namespace hal { namespace cuda {

device::device(CUdevice cudaDevice, coherence_domain &coherenceDomain) :
    Parent(coherenceDomain),
    cudaDevice_(cudaDevice)
{
    CUresult ret;
    int val;

    TRACE(GLOBAL, "Creating CUDA accelerator %d", cudaDevice_);
    ret = cuDeviceTotalMem(&memorySize_, cudaDevice_);
    CFATAL(ret == CUDA_SUCCESS, "Unable to initialize CUDA device %d", ret);
    ret = cuDeviceComputeCapability(&major_, &minor_, cudaDevice_);
    CFATAL(ret == CUDA_SUCCESS, "Unable to initialize CUDA device %d", ret);
    ret = cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_INTEGRATED, cudaDevice_);
    CFATAL(ret == CUDA_SUCCESS, "Unable to get attribute %d", ret);
    integrated_ = (val != 0);
}

aspace_t
device::create_address_space()
{
    CUcontext ctx, tmp;
    unsigned int flags = 0;
#if CUDA_VERSION >= 2020
    if(major_ >= 2 || (major_ == 1 && minor_ >= 1)) flags |= CU_CTX_MAP_HOST;
#else
    TRACE(LOCAL,"Host mapped memory not supported by the HW");
#endif
    CUresult ret = cuCtxCreate(&ctx, flags, cudaDevice_);
    if(ret != CUDA_SUCCESS)
        FATAL("Unable to create CUDA context %d", ret);
    ret = cuCtxPopCurrent(&tmp);
    ASSERTION(ret == CUDA_SUCCESS);

    return aspace_t(ctx, *this);
}

static void
set_address_space(const aspace_t &aspace)
{
    CUresult res = cuCtxSetCurrent(aspace());
    ASSERTION(res == CUDA_SUCCESS, "Error setting the context");
}

stream_t
device::create_stream(aspace_t &aspace)
{
    set_address_space(aspace); 
    CUstream stream;
    CUresult ret = cuStreamCreate(&stream, 0);
    CFATAL(ret == CUDA_SUCCESS, "Unable to create CUDA stream");

    return stream_t(stream, aspace);
}

event_t
device::copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream)
{
    CUresult res;

    event_t ret(stream);

    ret.begin(stream);
    res = cuMemcpyHtoD(dst.get(), src, count);
    ret.end(stream);

    return ret;
}

event_t
device::copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream)
{
    CUresult res;

    event_t ret(stream);

    ret.begin(stream);
    res = cuMemcpyDtoH(dst, src.get(), count);
    ret.end(stream);

    return ret;
}

event_t
device::copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream)
{
    CUresult res;

    event_t ret(stream);

    ret.begin(stream);
    res = cuMemcpyDtoD(dst.get(), src.get(), count);
    ret.end(stream);

    return ret;
}

async_event_t
device::copy_async(accptr_t dst, hostptr_t src, size_t count, stream_t &stream)
{
    CUresult res;

    async_event_t ret(stream);

    ret.begin(stream);
    res = cuMemcpyHtoDAsync(dst.get(), src, count, stream());
    ret.end(stream);

    return ret;
}

async_event_t
device::copy_async(hostptr_t dst, accptr_t src, size_t count, stream_t &stream)
{
    CUresult res;

    async_event_t ret(stream);

    ret.begin(stream);
    res = cuMemcpyDtoHAsync(dst, src.get(), count, stream());
    ret.end(stream);

    return ret;
}

async_event_t
device::copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream)
{
    CUresult res;

    async_event_t ret(stream);

    ret.begin(stream);
    res = cuMemcpyDtoDAsync(dst.get(), src.get(), count, stream());
    ret.end(stream);

    return ret;
}

gmacError_t
device::sync(async_event_t &event)
{
    CUresult res;

    res = cuEventSynchronize(event.eventEnd_);

    return cuda::error(res);
}

gmacError_t
device::sync(stream_t &stream)
{
    CUresult res;

    event_t ret(stream);

    ret.begin(stream);
    res = cuStreamSynchronize(stream());
    ret.end(stream);

    return cuda::error(res);
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

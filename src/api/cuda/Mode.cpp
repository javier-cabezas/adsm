#include "Mode.h"
#include "Context.h"
#include "Accelerator.h"

namespace gmac { namespace gpu {

Mode::Mode(Accelerator *acc) :
    gmac::Mode(acc),
    __acc(acc),
    hostBuffer(paraver::LockIoHost, this),
    deviceBuffer(paraver::LockIoDevice, this),
    __call(dim3(0), dim3(0), 0, 0)
{
#ifdef USE_MULTI_CONTEXT
    __ctx = __acc->createContext();
#endif
}

void Mode::setupStreams()
{
    CUresult ret;
    ret = cuStreamCreate(&streamLaunch, 0);
    cfatal(ret == CUDA_SUCCESS, "Unable to create CUDA stream %d", ret);
    ret = cuStreamCreate(&streamToDevice, 0);
    cfatal(ret == CUDA_SUCCESS, "Unable to create CUDA stream %d", ret);
    ret = cuStreamCreate(&streamToHost, 0);
    cfatal(ret == CUDA_SUCCESS, "Unable to create CUDA stream %d", ret);
    ret = cuStreamCreate(&streamDevice, 0);
    cfatal(ret == CUDA_SUCCESS, "Unable to create CUDA stream %d", ret);
}

void Mode::cleanStreams()
{
    CUresult ret;
    ret = cuStreamDestroy(streamLaunch);
    cfatal(ret == CUDA_SUCCESS, "Error destroying CUDA streams: %d", ret);
    ret = cuStreamDestroy(streamToDevice);
    cfatal(ret == CUDA_SUCCESS, "Error destroying CUDA streams: %d", ret);
    ret = cuStreamDestroy(streamToHost);
    cfatal(ret == CUDA_SUCCESS, "Error destroying CUDA streams: %d", ret);
    ret = cuStreamDestroy(streamDevice);
    cfatal(ret == CUDA_SUCCESS, "Error destroying CUDA streams: %d", ret);
}

gmacError_t Mode::syncStream(CUstream stream)
{
    CUresult ret = CUDA_SUCCESS;

    switchIn();
    while ((ret = cuStreamQuery(stream)) == CUDA_ERROR_NOT_READY) {
        switchOut();
        // TODO: add delay here
        switchIn();
    }
    switchOut();
    popEventState(paraver::Accelerator, 0x10000000 + _id);

    if (ret == CUDA_SUCCESS) { trace("Sync: success"); }
    else { trace("Sync: error: %d", ret); }

    return Accelerator::error(ret);
}

gmacError_t Mode::copyToDevice(void *dev, const void *host, size_t size)
{
    if(size == 0) return gmacSuccess; /* Fast path */
    /* In case there is no page-locked memory available, use the slow path */
    if(deviceBuffer.ptr() == NULL)
        return gmac::Mode::copyToDevice(dev, host, size);

    gmacError_t ret = gmacSuccess;
    size_t offset = 0;
    deviceBuffer.lock();
    switchIn();
    while(offset < size) {
        if(deviceBuffer.ready() == false) ret = __acc->syncStream(streamDevice);
        if(ret != gmacSuccess) break;
        size_t len = hostBuffer.size();
        if((size - offset) < hostBuffer.size()) len = size - offset;
        memcpy(deviceBuffer.ptr(), (uint8_t *)host + offset, len);
        ret = __acc->copyToDeviceAsync((uint8_t *)dev + offset, deviceBuffer.ptr(), len, streamDevice);
        if(ret != gmacSuccess) break;
        offset += len;
        deviceBuffer.busy();
    }
    switchOut();
    deviceBuffer.unlock();
    return ret;
}

gmacError_t Mode::copyToHost(void *host, const void *device, size_t size)
{
    if(size == 0) return gmacSuccess;
    if(hostBuffer.ptr() == NULL)
        return gmac::Mode::copyToHost(host, device, size);

    gmacError_t ret = gmacSuccess;
    size_t offset = size;
    hostBuffer.lock();
    switchIn();
    while(offset < size) {
        assert(hostBuffer.ready() == true);
        if(ret != gmacSuccess) break;
        size_t len = hostBuffer.size();
        if((size - offset) < hostBuffer.size()) len = size - offset;
        ret = __acc->copyToHost(hostBuffer.ptr(), device, len);
        if(ret != gmacSuccess) break;
        memcpy((uint8_t *)host + offset, hostBuffer.ptr(), len);
        offset += len;
    }

    switchOut();
    hostBuffer.unlock();
    return ret;
}

gmacError_t Mode::hostAlloc(void **addr, size_t size)
{
    switchIn();
#if CUDART_VERSION >= 2020
    CUresult ret = cuMemHostAlloc(addr, size, CU_MEMHOSTALLOC_PORTABLE);
#else
    CUresult ret = cuMemAllocHost(addr, size);
#endif
    switchOut();
    return Accelerator::error(ret);
}

gmacError_t Mode::hostFree(void *addr)
{
    switchIn();
    CUresult r = cuMemFreeHost(addr);
    switchOut();
    return Accelerator::error(r);
}


const Variable *Mode::constant(gmacVariable_t key) const
{
    return dynamic_cast<Context *>(__context)->constant(key);
}


const Variable *Mode::variable(gmacVariable_t key) const
{
    return dynamic_cast<Context *>(__context)->variable(key);
}

const Texture *Mode::texture(gmacTexture_t key) const
{
    return dynamic_cast<Context *>(__context)->texture(key);
}


}}

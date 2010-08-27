#include "Mode.h"

namespace gmac { namespace gpu {

Mode::Mode() :
    __hostBuffer(paraver::LockIoHost, this),
    __deviceBuffer(paraver::LockIoDevice, this)
{
#ifdef USE_MULTI_CONTEXT
    __ctx = __acc->createContext();
#endif
}

void Mode::setupStreams()
{
    CUresult ret;
    ret = cuStreamCreate(&__exe, 0);
    cfatal(ret == CUDA_SUCCESS, "Unable to create CUDA stream %d", ret);
    ret = cuStreamCreate(&__device, 0);
    cfatal(ret == CUDA_SUCCESS, "Unable to create CUDA stream %d", ret);
    ret = cuStreamCreate(&__host, 0);
    cfatal(ret == CUDA_SUCCESS, "Unable to create CUDA stream %d", ret);
    ret = cuStreamCreate(&__internal, 0);
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
    if(__deviceBuffer.ptr() == NULL)
        return gmac::Mode::copyToDevice(dev, host, size);

    gmacError_t ret = gmacSuccess;
    size_t offset = 0;
    __device.lock();
    switchIn();
    while(offset < size) {
        if(__deviceBuffer.ready() == false) ret = acc->syncStream(__device);
        if(ret != gmacSuccess) break;
        size_t len = __hostBuffer.size();
        if((size - offset) < __hostBuffer.size()) len = size - offset;
        memcpy(__deviceBuffer.ptr(), (uint8_t *)host + offset, len);
        ret = __acc->copyToDevice((uint8_t *)dev + offset, __deviceBuffer.ptr(), len);
        if(ret != gmacSuccess) break;
        offset += len;
        __deviceBuffer.busy();
    }
    switchOut();
    __device.unlock();
    return ret;
}

gmacError_t Mode::copyToHost(void *host, const void *device, size_t size)
{
    if(size == 0) return gmacSuccess;
    if(__hostBuffer.ptr() == NULL)
        return gmac::Mode::copyToHost(host, device, size);

    size_t offset = size;
    __host.lock();
    switchIn();
    while(offset < size) {
        assert(__hostBuffer.ready() == true);
        if(ret != gmacSuccess) break;
        size_t len = __hostBuffer.size();
        if((size - offset) < __hostBuffer.size()) len = size - offset;
        ret = __acc->copyToHost(__hostBuffer.ptr(), device, len);
        if(ret != gmacSuccess) break;
        memcpy((uint8_t *)host + offset, __hostBuffer.ptr(), len);
        offset += len;
    }

    switchOut();
    __hostUnlock;
    return ret;
}


}}

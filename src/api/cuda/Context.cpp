#include "Context.h"
#include "Mode.h"

#include <config.h>

#include <memory/Manager.h>
#include <gmac/init.h>

namespace gmac { namespace gpu {

Context::AddressMap Context::hostMem;
void * Context::FatBin;

Context::Context(Accelerator *acc, Mode *mode) :
    gmac::Context(acc),
    inputBuffer(mode, paramBufferPageLockedSize * paramPageSize),
    outputBuffer(mode, paramBufferPageLockedSize * paramPageSize),
    __call(dim3(0), dim3(0), 0, NULL)
{
    modules = ModuleDescriptor::createModules(*this);
    setupStreams();
}

Context::~Context()
{
    trace("Remove Accelerator context [%p]", this);
    ModuleVector::const_iterator m;
    for(m = modules.begin(); m != modules.end(); m++) {
        delete (*m);
    }
    modules.clear();
    cleanStreams();
}

void Context::setupStreams()
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

void Context::cleanStreams()
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

gmacError_t Context::syncStream(CUstream stream)
{
    CUresult ret = CUDA_SUCCESS;

    while ((ret = cuStreamQuery(stream)) == CUDA_ERROR_NOT_READY) {
        // TODO: add delay here
    }
    popEventState(paraver::Accelerator, 0x10000000 + _id);

    if (ret == CUDA_SUCCESS) { trace("Sync: success"); }
    else { trace("Sync: error: %d", ret); }

    return Accelerator::error(ret);
}

gmacError_t Context::copyToDevice(void *dev, const void *host, size_t size)
{
    if(size == 0) return gmacSuccess; /* Fast path */
    /* In case there is no page-locked memory available, use the slow path */
    if(outputBuffer.addr() == NULL)
        return gmac::Context::copyToDevice(dev, host, size);

    gmacError_t ret = gmacSuccess;
    size_t offset = 0;
    outputBuffer.lock();
    while(offset < size) {
        if(outputBuffer.state() == IOBuffer::Idle)
            ret = acc->syncStream(streamToDevice);
        if(ret != gmacSuccess) break;
        size_t len = outputBuffer.size();
        if((size - offset) < outputBuffer.size()) len = size - offset;
        memcpy(outputBuffer.addr(), (uint8_t *)host + offset, len);
        ret = acc->copyToDeviceAsync((uint8_t *)dev + offset, outputBuffer.addr(), len, streamToDevice);
        if(ret != gmacSuccess) break;
        offset += len;
    }
    outputBuffer.unlock();
    return ret;
}

gmacError_t Context::copyToHost(void *host, const void *device, size_t size)
{
    if(size == 0) return gmacSuccess;
    if(inputBuffer.addr() == NULL)
        return gmac::Context::copyToHost(host, device, size);

    gmacError_t ret = gmacSuccess;
    size_t offset = size;
    inputBuffer.lock();
    while(offset < size) {
        if(ret != gmacSuccess) break;
        size_t len = inputBuffer.size();
        if((size - offset) < inputBuffer.size()) len = size - offset;
        ret = acc->copyToHost(inputBuffer.addr(), device, len);
        if(ret != gmacSuccess) break;
        memcpy((uint8_t *)host + offset, inputBuffer.addr(), len);
        offset += len;
    }
    inputBuffer.unlock();
    return ret;
}

gmacError_t Context::bufferToDevice(IOBuffer *buffer, void *addr, size_t len)
{
    size_t bytes = (len < buffer->size()) ? len : buffer->size();
    return acc->copyToDeviceAsync(addr, buffer->addr(), bytes, streamToDevice);
}

gmacError_t Context::waitDevice()
{
    return syncStream(streamToDevice);
}

gmacError_t Context::bufferToHost(IOBuffer *buffer, void *addr, size_t len)
{
    size_t bytes = (len < buffer->size()) ? len : buffer->size();
    return acc->copyToHostAsync(buffer->addr(), addr, bytes, streamToHost);
}

gmacError_t Context::waitHost()
{
    return syncStream(streamToHost);
}

}}

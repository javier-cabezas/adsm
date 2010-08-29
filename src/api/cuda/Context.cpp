#include "Context.h"
#include "Mode.h"

#include <config.h>

#include <memory/Manager.h>
#include <gmac/init.h>

namespace gmac { namespace gpu {

Buffer::Buffer(paraver::LockName name, Mode *mode) :
    util::Lock(name),
    mode(mode),
    __ready(true)
{
    __size = paramBufferPageLockedSize * paramPageSize;

    gmacError_t ret = mode->hostAlloc(&buffer, __size);
    if (ret == gmacSuccess) {
        trace("Using page locked memory: %zd", __size);
    } else {
        trace("Not using page locked memoryError %d");
        buffer = NULL;
    }
}

Buffer::~Buffer()
{
    if(buffer == NULL) return;
    gmacError_t ret = mode->hostFree(buffer);
    if(ret != gmacSuccess) warning("Error release mode buffer. I will continue anyway");
}


Context::AddressMap Context::hostMem;
void * Context::FatBin;

Context::Context(Accelerator *acc, Mode *mode) :
    gmac::Context(acc),
    inputBuffer(paraver::LockIoDevice, mode),
    outputBuffer(paraver::LockIoHost, mode),
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
    if(outputBuffer.ptr() == NULL)
        return gmac::Context::copyToDevice(dev, host, size);

    gmacError_t ret = gmacSuccess;
    size_t offset = 0;
    outputBuffer.lock();
    while(offset < size) {
        if(outputBuffer.ready() == false) ret = acc->syncStream(streamDevice);
        if(ret != gmacSuccess) break;
        size_t len = outputBuffer.size();
        if((size - offset) < outputBuffer.size()) len = size - offset;
        memcpy(outputBuffer.ptr(), (uint8_t *)host + offset, len);
        ret = acc->copyToDeviceAsync((uint8_t *)dev + offset, outputBuffer.ptr(), len, streamDevice);
        if(ret != gmacSuccess) break;
        offset += len;
        outputBuffer.busy();
    }
    outputBuffer.unlock();
    return ret;
}

gmacError_t Context::copyToHost(void *host, const void *device, size_t size)
{
    if(size == 0) return gmacSuccess;
    if(inputBuffer.ptr() == NULL)
        return gmac::Context::copyToHost(host, device, size);

    gmacError_t ret = gmacSuccess;
    size_t offset = size;
    inputBuffer.lock();
    while(offset < size) {
        assert(inputBuffer.ready() == true);
        if(ret != gmacSuccess) break;
        size_t len = inputBuffer.size();
        if((size - offset) < inputBuffer.size()) len = size - offset;
        ret = acc->copyToHost(inputBuffer.ptr(), device, len);
        if(ret != gmacSuccess) break;
        memcpy((uint8_t *)host + offset, inputBuffer.ptr(), len);
        offset += len;
    }

    inputBuffer.unlock();
    return ret;
}

}}

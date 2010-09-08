#include "Context.h"
#include "Mode.h"

#include <config.h>

#include <gmac/init.h>
#include <memory/Manager.h>
#include <trace/Thread.h>

namespace gmac { namespace cuda {

Context::AddressMap Context::hostMem;
void * Context::FatBin;

Context::Context(Accelerator *acc, Mode *mode) :
    gmac::Context(acc, mode->id()),
    acc(acc),
    buffer(paramBufferPageLockedSize * paramPageSize),
    _call(dim3(0), dim3(0), 0, NULL)
{
    setupStreams();
    _call.stream(_streamLaunch);
}

Context::~Context()
{
    trace("Remove Accelerator context [%p]", this);
    cleanStreams();
}

void Context::setupStreams()
{
    CUresult ret;
    ret = cuStreamCreate(&_streamLaunch, 0);
    cfatal(ret == CUDA_SUCCESS, "Unable to create CUDA _stream %d", ret);
    ret = cuStreamCreate(&_streamToDevice, 0);
    cfatal(ret == CUDA_SUCCESS, "Unable to create CUDA _stream %d", ret);
    ret = cuStreamCreate(&_streamToHost, 0);
    cfatal(ret == CUDA_SUCCESS, "Unable to create CUDA _stream %d", ret);
    ret = cuStreamCreate(&_streamDevice, 0);
    cfatal(ret == CUDA_SUCCESS, "Unable to create CUDA _stream %d", ret);
}

void Context::cleanStreams()
{
    CUresult ret;
    ret = cuStreamDestroy(_streamLaunch);
    cfatal(ret == CUDA_SUCCESS, "Error destroying CUDA _streams: %d", ret);
    ret = cuStreamDestroy(_streamToDevice);
    cfatal(ret == CUDA_SUCCESS, "Error destroying CUDA _streams: %d", ret);
    ret = cuStreamDestroy(_streamToHost);
    cfatal(ret == CUDA_SUCCESS, "Error destroying CUDA _streams: %d", ret);
    ret = cuStreamDestroy(_streamDevice);
    cfatal(ret == CUDA_SUCCESS, "Error destroying CUDA _streams: %d", ret);
}

gmacError_t Context::syncStream(CUstream _stream)
{
    CUresult ret = CUDA_SUCCESS;

    gmac::trace::Thread::io();
    while ((ret = cuStreamQuery(_stream)) == CUDA_ERROR_NOT_READY) {
        // TODO: add delay here
    }
    gmac::trace::Thread::resume();

    if (ret == CUDA_SUCCESS) { trace("Sync: success"); }
    else { trace("Sync: error: %d", ret); }

    return Accelerator::error(ret);
}

gmacError_t Context::waitForBuffer(IOBuffer *buffer)
{
    gmacError_t ret = gmacErrorUnknown;
    switch(buffer->state()) {
        case IOBuffer::Idle: return gmacSuccess;
        case IOBuffer::ToDevice:
            ret = syncStream(_streamToDevice);
            toDeviceBuffer->state(IOBuffer::Idle);
            assert(buffer->state() == IOBuffer::Idle);
            break;
        case IOBuffer::ToHost:
            ret = syncStream(_streamToHost);
            toHostBuffer->state(IOBuffer::Idle);
            assert(buffer->state() == IOBuffer::Idle);
            break;
    };
    return ret;
}

gmacError_t Context::copyToDevice(void *dev, const void *host, size_t size)
{
    trace("Transferring %zd bytes from host %p to device %p", size, host, dev);
    if(size == 0) return gmacSuccess; /* Fast path */
    /* In case there is no page-locked memory available, use the slow path */
    if(buffer.isPinned() == false) {
        trace("Not using pinned memory for transfer");
        return gmac::Context::copyToDevice(dev, host, size);
    }

    gmacError_t ret = gmacSuccess;
    size_t offset = 0;
    buffer.lock();
    while(offset < size) {
        if(buffer.state() != IOBuffer::Idle)
            ret = acc->syncStream(_streamToDevice);
        if(ret != gmacSuccess) break;
        size_t len = buffer.size();
        if((size - offset) < buffer.size()) len = size - offset;
        memcpy(buffer.addr(), (uint8_t *)host + offset, len);
        buffer.state(IOBuffer::ToDevice);
        ret = acc->copyToDeviceAsync((uint8_t *)dev + offset, buffer.addr(), len, _streamToDevice);
        if(ret != gmacSuccess) break;
        offset += len;
    }
    buffer.unlock();
    return ret;
}

gmacError_t Context::copyToHost(void *host, const void *device, size_t size)
{
    trace("Transferring %zd bytes from device %p to host %p", size, device, host);

    if(size == 0) return gmacSuccess;
    if(buffer.isPinned() == false)
        return gmac::Context::copyToHost(host, device, size);

    gmacError_t ret = gmacSuccess;
    size_t offset = 0;
    buffer.lock();
    assertion(buffer.state() == IOBuffer::Idle);
    while(offset < size) {
        if(ret != gmacSuccess) break;
        size_t len = buffer.size();
        if((size - offset) < buffer.size()) len = size - offset;
        buffer.state(IOBuffer::ToHost);
        ret = acc->copyToHostAsync(buffer.addr(), (uint8_t *)device + offset, len, _streamToHost);
        if(ret != gmacSuccess) break;
        ret = acc->syncStream(_streamToHost);
        if(ret != gmacSuccess) break;
        memcpy((uint8_t *)host + offset, buffer.addr(), len);
        offset += len;
    }
    buffer.state(IOBuffer::Idle);
    buffer.unlock();
    return ret;
}

gmacError_t Context::copyDevice(void *dst, const void *src, size_t size)
{
    gmacError_t ret = acc->copyDevice(dst, src, size);
    return ret;
}

gmacError_t Context::memset(void *addr, int c, size_t size)
{
    gmacError_t ret = acc->memset(addr, c, size);
    return ret;
}

gmac::KernelLaunch *Context::launch(gmac::Kernel *kernel)
{
    return kernel->launch(_call);
}

gmacError_t Context::sync()
{
    switch(buffer.state()) {
        case IOBuffer::ToHost: syncStream(_streamToHost); break;
        case IOBuffer::ToDevice: syncStream(_streamToDevice); break;
        case IOBuffer::Idle: break;
    }
    buffer.state(IOBuffer::Idle);
    return syncStream(_streamLaunch);
}

gmacError_t Context::bufferToDevice(IOBuffer *buffer, void *addr, size_t len)
{
    gmacError_t ret = waitForBuffer(buffer);
    if(ret != gmacSuccess) return ret;
    size_t bytes = (len < buffer->size()) ? len : buffer->size();
    buffer->state(IOBuffer::ToDevice);
    toDeviceBuffer = buffer;
    ret = acc->copyToDeviceAsync(addr, buffer->addr(), bytes, _streamToDevice);
    return ret;
}

gmacError_t Context::bufferToHost(IOBuffer *buffer, void *addr, size_t len)
{
    gmacError_t ret = waitForBuffer(buffer);
    if(ret != gmacSuccess) return ret;
    buffer->state(IOBuffer::ToHost);
    toHostBuffer = buffer;
    size_t bytes = (len < buffer->size()) ? len : buffer->size();
    ret = acc->copyToHostAsync(buffer->addr(), addr, bytes, _streamToHost);
    if(ret != gmacSuccess) return ret;
    ret = waitForBuffer(buffer);
    return ret;
}

}}

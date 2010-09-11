#include "Context.h"
#include "Mode.h"

#include <config.h>

#include "gmac/init.h"
#include "memory/Manager.h"
#include "trace/Thread.h"

namespace gmac { namespace cuda {

Context::AddressMap Context::_HostMem;
void * Context::_FatBin;

Context::Context(Accelerator *acc, Mode *mode) :
    gmac::Context(acc, mode->id()),
    _acc(acc),
    _buffer(NULL),
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

    if(_buffer != NULL) proc->destroyIOBuffer(_buffer);
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
            break;
        case IOBuffer::ToHost:
            ret = syncStream(_streamToHost);
            break;
    };

    buffer->state(IOBuffer::Idle);
    return ret;
}

gmacError_t Context::copyToDevice(void *dev, const void *host, size_t size)
{
    trace("Transferring %zd bytes from host %p to device %p", size, host, dev);
    if(size == 0) return gmacSuccess; /* Fast path */
    /* In case there is no page-locked memory available, use the slow path */
    if(_buffer == NULL) _buffer = proc->createIOBuffer(paramPageSize);
    if(_buffer == NULL) {
        trace("Not using pinned memory for transfer");
        return gmac::Context::copyToDevice(dev, host, size);
    }

    gmacError_t ret = waitForBuffer(_buffer);
    if(ret != gmacSuccess) return ret;
    size_t offset = 0;
    while(offset < size) {
        ret = waitForBuffer(_buffer);
        if(ret != gmacSuccess) break;
        size_t len = _buffer->size();
        if((size - offset) < _buffer->size()) len = size - offset;
        memcpy(_buffer->addr(), (uint8_t *)host + offset, len);
        _buffer->state(IOBuffer::ToDevice);
        ret = _acc->copyToDeviceAsync((uint8_t *)dev + offset, _buffer->addr(), len, _streamToDevice);
        if(ret != gmacSuccess) break;
        offset += len;
    }
    return ret;
}

gmacError_t Context::copyToHost(void *host, const void *device, size_t size)
{
    trace("Transferring %zd bytes from device %p to host %p", size, device, host);

    if(size == 0) return gmacSuccess;
    if(_buffer == NULL) _buffer = proc->createIOBuffer(paramPageSize);
    if(_buffer == NULL)
        return gmac::Context::copyToHost(host, device, size);

    gmacError_t ret = waitForBuffer(_buffer);
    if(ret != gmacSuccess) return ret;
    size_t offset = 0;
    while(offset < size) {
        size_t len = _buffer->size();
        if((size - offset) < _buffer->size()) len = size - offset;
        _buffer->state(IOBuffer::ToHost);
        ret = _acc->copyToHostAsync(_buffer->addr(), (uint8_t *)device + offset, len, _streamToHost);
        if(ret != gmacSuccess) break;
        ret = _acc->syncStream(_streamToHost);
        if(ret != gmacSuccess) break;
        memcpy((uint8_t *)host + offset, _buffer->addr(), len);
        offset += len;
    }
    return ret;
}

gmacError_t Context::copyDevice(void *dst, const void *src, size_t size)
{
    gmacError_t ret = _acc->copyDevice(dst, src, size);
    return ret;
}

gmacError_t Context::memset(void *addr, int c, size_t size)
{
    gmacError_t ret = _acc->memset(addr, c, size);
    return ret;
}

gmac::KernelLaunch *Context::launch(gmac::Kernel *kernel)
{
    return kernel->launch(_call);
}

gmacError_t Context::sync()
{
    if(_buffer != NULL) {
        switch(_buffer->state()) {
            case IOBuffer::ToHost: syncStream(_streamToHost); break;
            case IOBuffer::ToDevice: syncStream(_streamToDevice); break;
            case IOBuffer::Idle: break;
        }
       _buffer->state(IOBuffer::Idle);
    }
    return syncStream(_streamLaunch);
}

gmacError_t Context::bufferToDevice(void * dst, IOBuffer *buffer, size_t len, off_t off)
{
    gmacError_t ret = waitForBuffer(buffer);
    if(ret != gmacSuccess) return ret;
    size_t bytes = (len < buffer->size()) ? len : buffer->size();
    buffer->state(IOBuffer::ToDevice);
    ret = _acc->copyToDeviceAsync(dst, (char *) buffer->addr() + off, bytes, _streamToDevice);
    return ret;
}

gmacError_t Context::deviceToBuffer(IOBuffer *buffer, const void * src, size_t len, off_t off)
{
    gmacError_t ret = waitForBuffer(buffer);
    if(ret != gmacSuccess) return ret;
    buffer->state(IOBuffer::ToHost);
    size_t bytes = (len < buffer->size()) ? len : buffer->size();
    ret = _acc->copyToHostAsync((char *) buffer->addr() + off, src, bytes, _streamToHost);
    if(ret != gmacSuccess) return ret;
    ret = waitForBuffer(buffer);
    return ret;
}

}}

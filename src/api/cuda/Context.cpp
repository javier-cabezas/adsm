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
    setupCUstreams();
    _call.stream(_streamLaunch);
}

Context::~Context()
{
    trace("Remove Accelerator context [%p]", this);
    cleanCUstreams();
}

void Context::setupCUstreams()
{
    CUresult ret;
    _streamLaunch   = _acc->createCUstream();
    _streamToDevice = _acc->createCUstream();
    _streamToHost   = _acc->createCUstream();
    _streamDevice   = _acc->createCUstream();
}

void Context::cleanCUstreams()
{
    CUresult ret;
    _acc->destroyCUstream(_streamLaunch);
    _acc->destroyCUstream(_streamToDevice);
    _acc->destroyCUstream(_streamToHost);
    _acc->destroyCUstream(_streamDevice);

    if(_buffer != NULL) proc->destroyIOBuffer(_buffer);
}

gmacError_t Context::syncCUstream(CUstream _stream)
{
    CUresult ret = CUDA_SUCCESS;

    gmac::trace::Thread::io();
    while ((ret = _acc->queryCUstream(_stream)) == CUDA_ERROR_NOT_READY) {
        // TODO: add delay here
    }
    gmac::trace::Thread::resume();

    if (ret == CUDA_SUCCESS) { trace("Sync: success"); }
    else { trace("Sync: error: %d", ret); }

    return Accelerator::error(ret);
}

gmacError_t Context::waitForBuffer(IOBuffer &buffer)
{
    gmacError_t ret = gmacErrorUnknown;
    switch(buffer.state()) {
        case IOBuffer::Idle: return gmacSuccess;
        case IOBuffer::ToDevice:
            ret = syncCUstream(_streamToDevice);
            break;
        case IOBuffer::ToHost:
            ret = syncCUstream(_streamToHost);
            break;
    };

    buffer.state(IOBuffer::Idle);
    return ret;
}

gmacError_t Context::copyToDevice(void *dev, const void *host, size_t size)
{
    trace("Transferring %zd bytes from host %p to device %p", size, host, dev);
    trace::Function::start("Context", "copyToDevice");
    if(size == 0) return gmacSuccess; /* Fast path */
    /* In case there is no page-locked memory available, use the slow path */
    if(_buffer == NULL) _buffer = proc->createIOBuffer(paramPageSize);
    if(_buffer == NULL) {
        trace("Not using pinned memory for transfer");
        trace::Function::end("Context");
        return gmac::Context::copyToDevice(dev, host, size);
    }

    gmacError_t ret = waitForBuffer(*_buffer);
    if(ret != gmacSuccess) { trace::Function::end("Context"); return ret; }
    size_t offset = 0;
    while(offset < size) {
        ret = waitForBuffer(*_buffer);
        if(ret != gmacSuccess) break;
        size_t len = _buffer->size();
        if((size - offset) < _buffer->size()) len = size - offset;
        trace::Function::start("Context", "memcpyToDevice");
        memcpy(_buffer->addr(), (uint8_t *)host + offset, len);
        trace::Function::end("Context");
        assertion(len <= paramPageSize);
        _buffer->state(IOBuffer::ToDevice);
        ret = _acc->copyToDeviceAsync((uint8_t *)dev + offset, _buffer->addr(), len, _streamToDevice);
        assertion(ret == gmacSuccess);
        if(ret != gmacSuccess) break;
        offset += len;
    }
    trace::Function::end("Context");
    return ret;
}

gmacError_t Context::copyToHost(void *host, const void *device, size_t size)
{
    trace("Transferring %zd bytes from device %p to host %p", size, device, host);
    trace::Function::start("Context", "copyToHost");
    if(size == 0) return gmacSuccess;
    if(_buffer == NULL) _buffer = proc->createIOBuffer(paramPageSize);
    if(_buffer == NULL) {
        trace::Function::end("Context");
        return gmac::Context::copyToHost(host, device, size);
    }

    gmacError_t ret = waitForBuffer(*_buffer);
    if(ret != gmacSuccess) { trace::Function::end("Context"); return ret; }
    size_t offset = 0;
    while(offset < size) {
        size_t len = _buffer->size();
        if((size - offset) < _buffer->size()) len = size - offset;
        _buffer->state(IOBuffer::ToHost);
        ret = _acc->copyToHostAsync(_buffer->addr(), (uint8_t *)device + offset, len, _streamToHost);
        if(ret != gmacSuccess) break;
        ret = _acc->syncCUstream(_streamToHost);
        if(ret != gmacSuccess) break;
        trace::Function::start("Context", "memcpyToHost");
        memcpy((uint8_t *)host + offset, _buffer->addr(), len);
        trace::Function::end("Context");
        offset += len;
    }
    trace::Function::end("Context");
    return ret;
}

gmacError_t Context::copyDevice(void *dst, const void *src, size_t size)
{
    trace::Function::start("Context", "copyDevice");
    gmacError_t ret = _acc->copyDevice(dst, src, size);
    trace::Function::end("Context");
    return ret;
}

gmacError_t Context::memset(void *addr, int c, size_t size)
{
    trace::Function::start("Context", "memset");
    gmacError_t ret = _acc->memset(addr, c, size);
    trace::Function::end("Context");
    return ret;
}

gmac::KernelLaunch *Context::launch(gmac::Kernel *kernel)
{
    trace::Function::start("Context", "launch");
    gmac::trace::Thread::run(id);
    gmac::KernelLaunch *ret = kernel->launch(_call);
    trace::Function::end("Context");
    return ret;
}

gmacError_t Context::sync()
{
    trace::Function::start("Context", "sync");
    if(_buffer != NULL) {
        switch(_buffer->state()) {
            case IOBuffer::ToHost: syncCUstream(_streamToHost); break;
            case IOBuffer::ToDevice: syncCUstream(_streamToDevice); break;
            case IOBuffer::Idle: break;
        }
       _buffer->state(IOBuffer::Idle);
    }
    gmacError_t ret = syncCUstream(_streamLaunch);
    gmac::trace::Thread::resume(id);
    trace::Function::end("Context");
    return ret;
}

gmacError_t Context::bufferToDevice(void * dst, IOBuffer &buffer, size_t len, off_t off)
{
    trace::Function::start("Context", "bufferToDevice");
    gmacError_t ret = waitForBuffer(buffer);
    if(ret != gmacSuccess) { trace::Function::end("Context"); return ret; }
    size_t bytes = (len < buffer.size()) ? len : buffer.size();
    buffer.state(IOBuffer::ToDevice);
    ret = _acc->copyToDeviceAsync(dst, (char *) buffer.addr() + off, bytes, _streamToDevice);
    trace::Function::end("Context");
    return ret;
}

gmacError_t Context::deviceToBuffer(IOBuffer &buffer, const void * src, size_t len, off_t off)
{
    trace::Function::start("Context", "deviceToBuffer");
    gmacError_t ret = waitForBuffer(buffer);
    if(ret != gmacSuccess) { trace::Function::end("Context"); return ret; }
    buffer.state(IOBuffer::ToHost);
    size_t bytes = (len < buffer.size()) ? len : buffer.size();
    ret = _acc->copyToHostAsync((char *) buffer.addr() + off, src, bytes, _streamToHost);
    if(ret != gmacSuccess) { trace::Function::end("Context"); return ret; }
    ret = waitForBuffer(buffer);
    trace::Function::end("Context");
    return ret;
}

}}

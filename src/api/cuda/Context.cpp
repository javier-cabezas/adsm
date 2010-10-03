#include "Context.h"
#include "Mode.h"

#include <config.h>

#include "gmac/init.h"
#include "memory/Manager.h"
#include "trace/Thread.h"

namespace gmac { namespace cuda {

Context::AddressMap Context::HostMem_;
void * Context::FatBin_;

Context::Context(Accelerator &acc, Mode &mode) :
    gmac::Context(acc, mode.id()),
    buffer_(NULL),
    call_(dim3(0), dim3(0), 0, NULL)
{
    setupCUstreams();
    call_.stream(streamLaunch_);
}

Context::~Context()
{
    //if(_buffer != NULL) delete _buffer;
    cleanCUstreams();
}

void Context::setupCUstreams()
{
    CUresult ret;
    streamLaunch_   = accelerator().createCUstream();
    streamToDevice_ = accelerator().createCUstream();
    streamToHost_   = accelerator().createCUstream();
    streamDevice_   = accelerator().createCUstream();
}

void Context::cleanCUstreams()
{
    CUresult ret;
    accelerator().destroyCUstream(streamLaunch_);
    accelerator().destroyCUstream(streamToDevice_);
    accelerator().destroyCUstream(streamToHost_);
    accelerator().destroyCUstream(streamDevice_);

    gmac::Process &proc = gmac::Process::getInstance();
    if(buffer_ != NULL) proc.destroyIOBuffer(buffer_);
}

gmacError_t Context::syncCUstream(CUstream _stream)
{
    CUresult ret = CUDA_SUCCESS;

    gmac::trace::Thread::io();
    while ((ret = accelerator().queryCUstream(_stream)) == CUDA_ERROR_NOT_READY) {
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
            ret = syncCUstream(streamToDevice_);
            break;
        case IOBuffer::ToHost:
            ret = syncCUstream(streamToHost_);
            break;
    }
    return ret;
}

gmacError_t Context::copyToDevice(void *dev, const void *host, size_t size)
{
    trace("Transferring %zd bytes from host %p to device %p", size, host, dev);
    trace::Function::start("Context", "copyToDevice");
    if(size == 0) return gmacSuccess; /* Fast path */
    /* In case there is no page-locked memory available, use the slow path */
    gmac::Process &proc = gmac::Process::getInstance();
    if(buffer_ == NULL) buffer_ = proc.createIOBuffer(paramPageSize);
    if(buffer_ == NULL) {
        trace("Not using pinned memory for transfer");
        trace::Function::end("Context");
        return gmac::Context::copyToDevice(dev, host, size);
    }

    gmacError_t ret = waitForBuffer(*buffer_);
    if(ret != gmacSuccess) { trace::Function::end("Context"); return ret; }
    size_t offset = 0;
    while(offset < size) {
        ret = waitForBuffer(*buffer_);
        if(ret != gmacSuccess) break;
        size_t len = buffer_->size();
        if((size - offset) < buffer_->size()) len = size - offset;
        trace::Function::start("Context", "memcpyToDevice");
        memcpy(buffer_->addr(), (uint8_t *)host + offset, len);
        trace::Function::end("Context");
        assertion(len <= paramPageSize);
        buffer_->toDevice();
        ret = accelerator().copyToDeviceAsync((uint8_t *)dev + offset, buffer_->addr(), len, streamToDevice_);
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
    gmac::Process &proc = gmac::Process::getInstance();
    if(buffer_ == NULL) buffer_ = proc.createIOBuffer(paramPageSize);
    if(buffer_ == NULL) {
        trace::Function::end("Context");
        return gmac::Context::copyToHost(host, device, size);
    }

    gmacError_t ret = waitForBuffer(*buffer_);
    if(ret != gmacSuccess) { trace::Function::end("Context"); return ret; }
    size_t offset = 0;
    while(offset < size) {
        size_t len = buffer_->size();
        if((size - offset) < buffer_->size()) len = size - offset;
        buffer_->toHost();
        ret = accelerator().copyToHostAsync(buffer_->addr(), (uint8_t *)device + offset, len, streamToHost_);
        if(ret != gmacSuccess) break;
        ret = accelerator().syncCUstream(streamToHost_);
        if(ret != gmacSuccess) break;
        trace::Function::start("Context", "memcpyToHost");
        memcpy((uint8_t *)host + offset, buffer_->addr(), len);
        trace::Function::end("Context");
        offset += len;
    }
    trace::Function::end("Context");
    return ret;
}

gmacError_t Context::copyDevice(void *dst, const void *src, size_t size)
{
    trace::Function::start("Context", "copyDevice");
    gmacError_t ret = accelerator().copyDevice(dst, src, size);
    trace::Function::end("Context");
    return ret;
}

gmacError_t Context::memset(void *addr, int c, size_t size)
{
    trace::Function::start("Context", "memset");
    gmacError_t ret = accelerator().memset(addr, c, size);
    trace::Function::end("Context");
    return ret;
}

gmac::KernelLaunch &Context::launch(gmac::Kernel &kernel)
{
    trace::Function::start("Context", "launch");
    gmac::trace::Thread::run(id_);
    gmac::KernelLaunch *ret = kernel.launch(call_);
    assertion(ret != NULL);
    trace::Function::end("Context");
    return *ret;
}

gmacError_t Context::sync()
{
    trace::Function::start("Context", "sync");
    if(buffer_ != NULL) {
    	buffer_->wait();
    }
    gmacError_t ret = syncCUstream(streamLaunch_);
    gmac::trace::Thread::resume(id_);
    trace::Function::end("Context");
    return ret;
}

gmacError_t Context::bufferToDevice(void * dst, IOBuffer &buffer, size_t len, off_t off)
{
    trace::Function::start("Context", "bufferToDevice");
    gmacError_t ret = waitForBuffer(buffer);
    if(ret != gmacSuccess) { trace::Function::end("Context"); return ret; }
    size_t bytes = (len < buffer.size()) ? len : buffer.size();
    buffer.toDevice();
    ret = accelerator().copyToDeviceAsync(dst, (char *) buffer.addr() + off, bytes, streamToDevice_);
    trace::Function::end("Context");
    return ret;
}

gmacError_t Context::deviceToBuffer(IOBuffer &buffer, const void * src, size_t len, off_t off)
{
    trace::Function::start("Context", "deviceToBuffer");
    gmacError_t ret = waitForBuffer(buffer);
    if(ret != gmacSuccess) { trace::Function::end("Context"); return ret; }
    buffer.toHost();
    size_t bytes = (len < buffer.size()) ? len : buffer.size();
    ret = accelerator().copyToHostAsync((char *) buffer.addr() + off, src, bytes, streamToHost_);
    if(ret != gmacSuccess) { trace::Function::end("Context"); return ret; }
    trace::Function::end("Context");
    return ret;
}

}}

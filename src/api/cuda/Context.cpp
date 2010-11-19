#include "Context.h"
#include "Mode.h"

#include "gmac/init.h"
#include "memory/Manager.h"
#include "trace/Tracer.h"

namespace __impl { namespace cuda {

Context::AddressMap Context::HostMem_;
void * Context::FatBin_;

Context::Context(Accelerator &acc, Mode &mode) :
    core::Context(acc, mode.id()),
    mode_(mode),
    buffer_(NULL),
    call_(dim3(0), dim3(0), 0, NULL)
{
    setupCUstreams();
    call_.stream(streamLaunch_);
}

Context::~Context()
{ 
    // Destroy context's private IOBuffer (if any)
    if(buffer_ != NULL) {
        TRACE(LOCAL,"Destroying I/O buffer");
    	mode_.destroyIOBuffer(buffer_);
    }

    cleanCUstreams();
}

void Context::setupCUstreams()
{
    streamLaunch_   = accelerator().createCUstream();
    streamToAccelerator_ = accelerator().createCUstream();
    streamToHost_   = accelerator().createCUstream();
    streamAccelerator_   = accelerator().createCUstream();
}

void Context::cleanCUstreams()
{
    accelerator().destroyCUstream(streamLaunch_);
    accelerator().destroyCUstream(streamToAccelerator_);
    accelerator().destroyCUstream(streamToHost_);
    accelerator().destroyCUstream(streamAccelerator_);
}

gmacError_t Context::syncCUstream(CUstream _stream)
{
    CUresult ret = CUDA_SUCCESS;

    trace::SetThreadState(trace::IO);
    while ((ret = accelerator().queryCUstream(_stream)) == CUDA_ERROR_NOT_READY) {
        // TODO: add delay here
    }
    trace::SetThreadState(trace::Running);

    if (ret == CUDA_SUCCESS) { TRACE(LOCAL,"Sync: success"); }
    else { TRACE(LOCAL,"Sync: error: %d", ret); }

    return Accelerator::error(ret);
}

gmacError_t Context::waitForBuffer(core::IOBuffer &buffer)
{
    gmacError_t ret = gmacErrorUnknown;
    switch(buffer.state()) {
        case core::IOBuffer::Idle: return gmacSuccess;
        case core::IOBuffer::ToAccelerator:
            ret = syncCUstream(streamToAccelerator_);
            break;
        case core::IOBuffer::ToHost:
            ret = syncCUstream(streamToHost_);
            break;
    }
    return ret;
}

gmacError_t Context::copyToAccelerator(void *dev, const void *host, size_t size)
{
    TRACE(LOCAL,"Transferring "FMT_SIZE" bytes from host %p to accelerator %p", size, host, dev);
    trace::EnterCurrentFunction();
    if(size == 0) return gmacSuccess; /* Fast path */
    /* In case there is no page-locked memory available, use the slow path */
    if(buffer_ == NULL) buffer_ = mode_.createIOBuffer(paramPageSize);
    if(buffer_ == NULL) {
        TRACE(LOCAL,"Not using pinned memory for transfer");
        trace::ExitCurrentFunction();
        return core::Context::copyToAccelerator(dev, host, size);
    }
    buffer_->wait();
    gmacError_t ret = gmacSuccess;
    size_t offset = 0;
    while(offset < size) {
        ret = buffer_->wait();
        if(ret != gmacSuccess) break;
        size_t len = buffer_->size();
        if((size - offset) < buffer_->size()) len = size - offset;
        trace::EnterCurrentFunction();
        memcpy(buffer_->addr(), (uint8_t *)host + offset, len);
        trace::ExitCurrentFunction();
        ASSERTION(len <= paramPageSize);
        buffer_->toAccelerator(mode_);
        ret = accelerator().copyToAcceleratorAsync((uint8_t *)dev + offset, buffer_->addr(), len, streamToAccelerator_);
        ASSERTION(ret == gmacSuccess);
        if(ret != gmacSuccess) break;
        offset += len;
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::copyToHost(void *host, const void *accAddr, size_t size)
{
    TRACE(LOCAL,"Transferring "FMT_SIZE" bytes from accelerator %p to host %p", size, accAddr, host);
    trace::EnterCurrentFunction();
    if(size == 0) return gmacSuccess;
    if(buffer_ == NULL) buffer_ = mode_.createIOBuffer(paramPageSize);
    if(buffer_ == NULL) {
        trace::ExitCurrentFunction();
        return core::Context::copyToHost(host, accAddr, size);
    }

    gmacError_t ret = buffer_->wait();
    buffer_->wait();
    if(ret != gmacSuccess) { trace::ExitCurrentFunction(); return ret; }
    size_t offset = 0;
    while(offset < size) {
        size_t len = buffer_->size();
        if((size - offset) < buffer_->size()) len = size - offset;
        buffer_->toHost(mode_);
        ret = accelerator().copyToHostAsync(buffer_->addr(), (uint8_t *)accAddr + offset, len, streamToHost_);
        ASSERTION(ret == gmacSuccess);
        if(ret != gmacSuccess) break;
        ret = buffer_->wait();
        if(ret != gmacSuccess) break;
        trace::EnterCurrentFunction();
        memcpy((uint8_t *)host + offset, buffer_->addr(), len);
        trace::ExitCurrentFunction();
        offset += len;
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::copyAccelerator(void *dst, const void *src, size_t size)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = accelerator().copyAccelerator(dst, src, size);
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::memset(void *addr, int c, size_t size)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = accelerator().memset(addr, c, size);
    trace::ExitCurrentFunction();
    return ret;
}

core::KernelLaunch &Context::launch(core::Kernel &kernel)
{
    trace::EnterCurrentFunction();
    trace::SetThreadState(THREAD_T(id_), trace::Running);
    core::KernelLaunch *ret = kernel.launch(call_);
    ASSERTION(ret != NULL);
    trace::ExitCurrentFunction();
    return *ret;
}

gmacError_t Context::sync()
{
    gmacError_t ret = gmacSuccess;
    trace::EnterCurrentFunction();	
    if(buffer_ != NULL) {
        waitForBuffer(*buffer_);
    }
    ret = syncCUstream(streamLaunch_);
    trace::SetThreadState(THREAD_T(id_), trace::Running);    
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::bufferToAccelerator(void * dst, core::IOBuffer &buffer, size_t len, off_t off)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = waitForBuffer(buffer);
    if(ret != gmacSuccess) { trace::ExitCurrentFunction(); return ret; }
    size_t bytes = (len < buffer.size()) ? len : buffer.size();
    buffer.toAccelerator(mode_);
    ret = accelerator().copyToAcceleratorAsync(dst, (char *) buffer.addr() + off, bytes, streamToAccelerator_);
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::acceleratorToBuffer(core::IOBuffer &buffer, const void * src, size_t len, off_t off)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = waitForBuffer(buffer);
    if(ret != gmacSuccess) { trace::ExitCurrentFunction(); return ret; }
    buffer.toHost(mode_);
    size_t bytes = (len < buffer.size()) ? len : buffer.size();
    ret = accelerator().copyToHostAsync((char *) buffer.addr() + off, src, bytes, streamToHost_);
    trace::ExitCurrentFunction();
    return ret;
}

}}

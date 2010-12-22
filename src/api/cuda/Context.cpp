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

gmacError_t Context::waitForEvent(CUevent e)
{
    gmacError_t ret = gmacErrorUnknown;
    ret = accelerator().syncCUevent(e);
    return ret;
}

gmacError_t Context::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size)
{
    TRACE(LOCAL,"Transferring "FMT_SIZE" bytes from host %p to accelerator %p", size, host, (void *) acc);
    trace::EnterCurrentFunction();
    if(size == 0) return gmacSuccess; /* Fast path */
    /* In case there is no page-locked memory available, use the slow path */
    if(buffer_ == NULL) buffer_ = static_cast<IOBuffer *>(mode_.createIOBuffer(paramPageSize));
    if(buffer_ == NULL) {
        TRACE(LOCAL,"Not using pinned memory for transfer");
        trace::ExitCurrentFunction();
        return core::Context::copyToAccelerator(acc, host, size);
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
        ::memcpy(buffer_->addr(), host + offset, len);
        trace::ExitCurrentFunction();
        ASSERTION(len <= paramPageSize);
        ret = accelerator().copyToAcceleratorAsync(acc + offset, *buffer_, 0, len, mode_, streamToAccelerator_);
        ASSERTION(ret == gmacSuccess);
        if(ret != gmacSuccess) break;
        offset += len;
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::copyToHost(hostptr_t host, const accptr_t acc, size_t size)
{
    TRACE(LOCAL,"Transferring "FMT_SIZE" bytes from accelerator %p to host %p", size, (void *) acc, host);
    trace::EnterCurrentFunction();
    if(size == 0) return gmacSuccess;
    if(buffer_ == NULL) buffer_ = static_cast<IOBuffer *>(mode_.createIOBuffer(paramPageSize));
    if(buffer_ == NULL) {
        trace::ExitCurrentFunction();
        return core::Context::copyToHost(host, acc, size);
    }

    gmacError_t ret = buffer_->wait();
    buffer_->wait();
    if(ret != gmacSuccess) { trace::ExitCurrentFunction(); return ret; }
    size_t offset = 0;
    while(offset < size) {
        size_t len = buffer_->size();
        if((size - offset) < buffer_->size()) len = size - offset;
        ret = accelerator().copyToHostAsync(*buffer_, 0, acc + offset, len, mode_, streamToHost_);
        ASSERTION(ret == gmacSuccess);
        if(ret != gmacSuccess) break;
        ret = buffer_->wait();
        if(ret != gmacSuccess) break;
        trace::EnterCurrentFunction();
        ::memcpy((uint8_t *)host + offset, buffer_->addr(), len);
        trace::ExitCurrentFunction();
        offset += len;
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::copyAccelerator(accptr_t dst, const accptr_t src, size_t size)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = accelerator().copyAccelerator(dst, src, size);
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::memset(accptr_t addr, int c, size_t size)
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
        buffer_->wait();
    }
    ret = syncCUstream(streamLaunch_);
    trace::SetThreadState(THREAD_T(id_), trace::Running);    
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::bufferToAccelerator(accptr_t dst, core::IOBuffer &_buffer, 
                                         size_t len, size_t off)
{
    trace::EnterCurrentFunction();
    IOBuffer &buffer = static_cast<IOBuffer &>(_buffer);
    gmacError_t ret = buffer.wait();
    if(ret != gmacSuccess) { trace::ExitCurrentFunction(); return ret; }
    ASSERTION(off + len <= buffer.size());
    ASSERTION(off >= 0);
    size_t bytes = (len < buffer.size()) ? len : buffer.size();
    ret = accelerator().copyToAcceleratorAsync(dst, buffer, off, bytes, mode_, streamToAccelerator_);
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::acceleratorToBuffer(core::IOBuffer &_buffer, const accptr_t src, 
                                         size_t len, size_t off)
{
    trace::EnterCurrentFunction();
    IOBuffer &buffer = static_cast<IOBuffer &>(_buffer);
    gmacError_t ret = buffer.wait();
    if(ret != gmacSuccess) { trace::ExitCurrentFunction(); return ret; }
    ASSERTION(off + len <= buffer.size());
    ASSERTION(off >= 0);
    size_t bytes = (len < buffer.size()) ? len : buffer.size();
    ret = accelerator().copyToHostAsync(buffer, off, src, bytes, mode_, streamToHost_);
    trace::ExitCurrentFunction();
    return ret;
}

}}

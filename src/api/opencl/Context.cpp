#include "Context.h"
#include "Mode.h"

#include "gmac/init.h"
#include "memory/Manager.h"
#include "trace/Tracer.h"

namespace __impl { namespace opencl {

Context::AddressMap Context::HostMem_;

Context::Context(Accelerator &acc, Mode &mode) :
    gmac::core::Context(acc, mode.id()),
    mode_(mode),
    buffer_(NULL)
{
    setupCLstreams();
}

Context::~Context()
{ 
    // Destroy context's private IOBuffer (if any)
    if(buffer_ != NULL) {
        TRACE(LOCAL,"Destroying I/O buffer");
    	mode_.destroyIOBuffer(buffer_);
    }

    cleanCLstreams();
}

void Context::setupCLstreams()
{
    Accelerator &acc = accelerator();
    streamLaunch_   = acc.createCLstream();
    TRACE(LOCAL, "cl_command_queue %p created for acc %p", streamLaunch_, &acc);
    streamToAccelerator_ = acc.createCLstream();
    TRACE(LOCAL, "cl_command_queue %p created for acc %p", streamToAccelerator_, &acc);
    streamToHost_   = acc.createCLstream();
    TRACE(LOCAL, "cl_command_queue %p created for acc %p", streamToHost_, &acc);
    streamAccelerator_   = acc.createCLstream();
    TRACE(LOCAL, "cl_command_queue %p created for acc %p", streamAccelerator_, &acc);
}

void Context::cleanCLstreams()
{
    Accelerator &acc = accelerator();
    acc.destroyCLstream(streamLaunch_);
    acc.destroyCLstream(streamToAccelerator_);
    acc.destroyCLstream(streamToHost_);
    acc.destroyCLstream(streamAccelerator_);
}

gmacError_t Context::syncCLstream(cl_command_queue stream)
{
    cl_int ret = CL_SUCCESS;

    trace::SetThreadState(trace::IO);
    Accelerator &acc = accelerator();
    TRACE(LOCAL, "Sync stream %p on accelerator %p", stream, &acc);
    ret = acc.syncCLstream(stream);
#if 0
    while ((ret = accelerator().queryCLstream(stream)) == CUDA_ERROR_NOT_READY) {
        // TODO: add delay here
    }
#endif
    trace::SetThreadState(trace::Running);

    if (ret == CL_SUCCESS) { TRACE(LOCAL,"Sync: success"); }
    else { TRACE(LOCAL,"Sync: error: %d", ret); }

    return Accelerator::error(ret);
}

gmacError_t Context::waitForEvent(cl_event e)
{
    gmacError_t ret = gmacErrorUnknown;
    ret = accelerator().syncCLevent(e);
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
    ptroff_t offset = 0;
    while(offset < size) {
        ret = buffer_->wait();
        if(ret != gmacSuccess) break;
        ptroff_t len = ptroff_t(buffer_->size());
        if((size - offset) < buffer_->size()) len = ptroff_t(size - offset);
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
    TRACE(LOCAL,"Transferring "FMT_SIZE" bytes from accelerator %p to host %p", size, acc.base_, host);
    trace::EnterCurrentFunction();
    if(size == 0) return gmacSuccess;
    if(buffer_ == NULL) buffer_ = static_cast<IOBuffer *>(mode_.createIOBuffer(paramPageSize));
    if(buffer_ == NULL) {
        TRACE(LOCAL,"Not using pinned memory for transfer");
        trace::ExitCurrentFunction();
        return core::Context::copyToHost(host, acc, size);
    }

    gmacError_t ret = buffer_->wait();
    buffer_->wait();
    if(ret != gmacSuccess) { trace::ExitCurrentFunction(); return ret; }
    ptroff_t offset = 0;
    while(offset < size) {
        ptroff_t len = ptroff_t(buffer_->size());
        if((size - offset) < buffer_->size()) len = ptroff_t(size - offset);
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
    ret = syncCLstream(streamLaunch_);
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

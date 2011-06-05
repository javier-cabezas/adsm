#include "api/opencl/IOBuffer.h"

#include "api/opencl/hpe/Context.h"
#include "api/opencl/hpe/Mode.h"
#include "api/opencl/hpe/Kernel.h"

#include "memory/Manager.h"

#include "trace/Tracer.h"

namespace __impl { namespace opencl { namespace hpe {

Context::Context(Mode &mode, cl_command_queue stream) :
    gmac::core::hpe::Context(mode, mode.id()),
    stream_(stream),
    buffer_(NULL)
{
}

Context::~Context()
{ 
    // Destroy context's private IOBuffer (if any)
    if(buffer_ != NULL) {
        TRACE(LOCAL,"Destroying I/O buffer");
    	dynamic_cast<Mode &>(mode_).destroyIOBuffer(*buffer_);
    }
}

Accelerator & Context::accelerator()
{
    return dynamic_cast<Accelerator &>(acc_);
}


gmacError_t Context::waitForEvent(cl_event e)
{
    gmacError_t ret = gmacErrorUnknown;
    ret = accelerator().syncCLevent(e);
    return ret;
}

gmacError_t Context::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size)
{
    TRACE(LOCAL,"Transferring "FMT_SIZE" bytes from host %p to accelerator %p", size, host, acc.get());
    trace::EnterCurrentFunction();
    if(size == 0) return gmacSuccess; /* Fast path */
    /* In case there is no page-locked memory available, use the slow path */
    if(buffer_ == NULL) buffer_ = &static_cast<IOBuffer &>(mode_.createIOBuffer(util::params::ParamBlockSize));
    if(buffer_->async() == false) {
        mode_.destroyIOBuffer(*buffer_);
        buffer_ = NULL;
        TRACE(LOCAL,"Not using pinned memory for transfer");
        trace::ExitCurrentFunction();
        return core::hpe::Context::copyToAccelerator(acc, host, size);
    }
    gmacError_t ret = buffer_->wait();;
    ptroff_t offset = 0;
    while(size_t(offset) < size) {
        ret = buffer_->wait();
        if(ret != gmacSuccess) break;
        ptroff_t len = ptroff_t(buffer_->size());
        if((size - offset) < buffer_->size()) len = ptroff_t(size - offset);
        trace::EnterCurrentFunction();
        ::memcpy(buffer_->addr(), host + offset, len);
        trace::ExitCurrentFunction();
        ASSERTION(size_t(len) <= util::params::ParamBlockSize);
        ret = accelerator().copyToAcceleratorAsync(acc + offset, *buffer_, 0, len, mode_, stream_);
        ASSERTION(ret == gmacSuccess);
        if(ret != gmacSuccess) break;
        offset += len;
    }
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::copyToHost(hostptr_t host, const accptr_t acc, size_t size)
{
    TRACE(LOCAL,"Transferring "FMT_SIZE" bytes from accelerator %p to host %p", size, acc.get(), host);
    trace::EnterCurrentFunction();
    if(size == 0) return gmacSuccess;
    if(buffer_ == NULL) buffer_ = &static_cast<IOBuffer &>(mode_.createIOBuffer(util::params::ParamBlockSize));
    if(buffer_->async() == false) {
        mode_.destroyIOBuffer(*buffer_);
        buffer_ = NULL;
        TRACE(LOCAL,"Not using pinned memory for transfer");
        trace::ExitCurrentFunction();
        return core::hpe::Context::copyToHost(host, acc, size);
    }
    gmacError_t ret = buffer_->wait();
    if(ret != gmacSuccess) { trace::ExitCurrentFunction(); return ret; }
    ptroff_t offset = 0;
    while(size_t(offset) < size) {
        ptroff_t len = ptroff_t(buffer_->size());
        if((size - offset) < buffer_->size()) len = ptroff_t(size - offset);
        ret = accelerator().copyToHostAsync(*buffer_, 0, acc + offset, len, mode_, stream_);
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

KernelLaunch &Context::launch(Kernel &kernel)
{
    trace::EnterCurrentFunction();
    KernelLaunch *ret = kernel.launch(dynamic_cast<Mode &>(mode_), stream_);
    ASSERTION(ret != NULL);
    trace::ExitCurrentFunction();
    return *ret;
}

gmacError_t Context::waitForCall(core::hpe::KernelLaunch &kl)
{
    trace::EnterCurrentFunction();
    KernelLaunch &launch = dynamic_cast<KernelLaunch &>(kl);
    gmacError_t ret = gmacSuccess;
    ret = waitForEvent(launch.getCLEvent());
    trace::SetThreadState(THREAD_T(id_), trace::Idle);
    trace::ExitCurrentFunction();
    return ret;
}

}}}

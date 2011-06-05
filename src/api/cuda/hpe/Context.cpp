#include "Context.h"
#include "Mode.h"

#include "hpe/init.h"
#include "memory/Manager.h"
#include "trace/Tracer.h"

namespace __impl { namespace cuda { namespace hpe {

Context::AddressMap Context::HostMem_;
void * Context::FatBin_;

Context::Context(Mode &mode, CUstream streamLaunch, CUstream streamToAccelerator,
                             CUstream streamToHost, CUstream streamAccelerator) :
    gmac::core::hpe::Context(mode, mode.id()),
    streamLaunch_(streamLaunch),
    streamToAccelerator_(streamToAccelerator),
    streamToHost_(streamToHost),
    streamAccelerator_(streamAccelerator),
    buffer_(NULL),
    call_(dim3(0), dim3(0), 0, NULL, NULL)
{
    call_ = KernelConfig(dim3(0), dim3(0), 0, NULL, streamLaunch_);
}

Context::~Context()
{ 
    // Destroy context's private IOBuffer (if any)
    if(buffer_ != NULL) {
        TRACE(LOCAL,"Destroying I/O buffer");
    	mode_.destroyIOBuffer(*buffer_);
    }
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
    if(buffer_ == NULL) buffer_ = &static_cast<IOBuffer &>(mode_.createIOBuffer(util::params::ParamBlockSize));
    if(buffer_->async() == false) {
        mode_.destroyIOBuffer(*buffer_);
        buffer_ = NULL;
        TRACE(LOCAL,"Not using pinned memory for transfer");
        trace::ExitCurrentFunction();
        return core::hpe::Context::copyToAccelerator(acc, host, size);
    }
    buffer_->waitFromCUDA();
    gmacError_t ret = gmacSuccess;
    size_t offset = 0;
    while(offset < size) {
        ret = buffer_->waitFromCUDA();
        if(ret != gmacSuccess) break;
        size_t len = buffer_->size();
        if((size - offset) < buffer_->size()) len = size - offset;
        trace::EnterCurrentFunction();
        ::memcpy(buffer_->addr(), host + offset, len);
        trace::ExitCurrentFunction();
        ASSERTION(len <= util::params::ParamBlockSize);
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
    if(buffer_ == NULL) buffer_ = &static_cast<IOBuffer &>(mode_.createIOBuffer(util::params::ParamBlockSize));
    if(buffer_->async() == false) {
        mode_.destroyIOBuffer(*buffer_);
        buffer_ = NULL;
        trace::ExitCurrentFunction();
        return core::hpe::Context::copyToHost(host, acc, size);
    }

    gmacError_t ret = buffer_->waitFromCUDA();
    if(ret != gmacSuccess) { trace::ExitCurrentFunction(); return ret; }
    size_t offset = 0;
    while(offset < size) {
        size_t len = buffer_->size();
        if((size - offset) < buffer_->size()) len = size - offset;
        ret = accelerator().copyToHostAsync(*buffer_, 0, acc + offset, len, mode_, streamToHost_);
        ASSERTION(ret == gmacSuccess);
        if(ret != gmacSuccess) break;
        ret = buffer_->waitFromCUDA();
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
    KernelLaunch *ret = kernel.launch(dynamic_cast<Mode &>(mode_), call_);
    ASSERTION(ret != NULL);
    trace::ExitCurrentFunction();
    return *ret;
}

gmacError_t Context::waitForCall(core::hpe::KernelLaunch &_launch)
{
    KernelLaunch &launch = dynamic_cast<KernelLaunch &>(_launch);
    gmacError_t ret = gmacSuccess;
    trace::EnterCurrentFunction();	
    ret = accelerator().syncCUevent(launch.getCUevent());
    trace::SetThreadState(THREAD_T(id_), trace::Idle);    
    trace::ExitCurrentFunction();
    return ret;
}

}}}

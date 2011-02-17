#ifndef GMAC_API_OPENCL_IOBUFFER_IMPL_H_
#define GMAC_API_OPENCL_IOBUFFER_IMPL_H_

#include "Tracer.h"

namespace __impl { namespace opencl {

inline
IOBuffer::IOBuffer(Mode &mode, cl_mem base, hostptr_t offset, size_t size, bool async) :
    gmac::core::IOBuffer(NULL, size, async), 
	base_(base),
	offset_(size_t(offset) - 0x1000),
	mode_(NULL), 
	started_(false)
{
    // Map the buffer to set the correct address
    addr_ = mode.hostMap(base_, offset_, size_);
    CFATAL(addr_ != NULL, "Unable to map I/O buffer in system memory");
}

inline cl_mem
IOBuffer::base() const 
{
    return base_;
}

inline size_t
IOBuffer::offset() const
{
    return offset_;
}

inline void
IOBuffer::toHost(cl_command_queue stream, Mode &mode)
{
    ASSERTION(started_ == false);

    gmacError_t ret = mode.hostUnmap(hostptr_t(addr_), base_, size_, stream);
    CFATAL(ret == gmacSuccess, "Unable to hand off I/O buffer to accelerator");

    state_  = ToHost;
    TRACE(LOCAL,"Buffer %p goes toHost", this); 
    mode_   = &mode;
}

inline void
IOBuffer::toAccelerator(cl_command_queue stream, Mode &mode)
{
    ASSERTION(started_ == false);

    gmacError_t ret = mode.hostUnmap(hostptr_t(addr_), base_, size_, stream);
    CFATAL(ret == gmacSuccess, "Unable to hand off I/O buffer to accelerator");

    state_  = ToAccelerator;
    TRACE(LOCAL,"Buffer %p goes toAccelerator", this);
    mode_   = &mode;
}

inline void
IOBuffer::started(cl_event event)
{
	TRACE(LOCAL,"Buffer %p starts", this);
    ASSERTION(started_ == false);
    ASSERTION(mode_ != NULL);

    addr_ = mode_->hostMap(base_, offset_, size_);
    CFATAL(addr_ != NULL, "Unable to map I/O buffer in system memory");

    event_ = event;
    started_ = true;
}

inline gmacError_t
IOBuffer::wait()
{
	TRACE(LOCAL,"Buffer %p waits: %d", this, state_ == Idle || started_ == true);
    ASSERTION(state_ == Idle || started_ == true);

    gmacError_t ret = gmacSuccess;

    if (state_ != Idle) {
        ASSERTION(mode_ != NULL);
        ASSERTION(event_ != NULL);

        trace::SetThreadState(trace::Wait);
        ret = mode_->waitForEvent(event_);
        trace::SetThreadState(trace::Running);
        ASSERTION(ret == gmacSuccess);
        if(state_ == ToHost) DataCommToHost(*mode_, event_, size_);
        else if(state_ == ToAccelerator) DataCommToAccelerator(*mode_, event_, size_);
        cl_int clret = clReleaseEvent(event_);
        ASSERTION(clret == CL_SUCCESS);
        TRACE(LOCAL,"Buffer %p goes Idle", this);


        state_ = Idle;
        event_ = NULL;
        mode_  = NULL;
        started_ = false;
    } else {
        ASSERTION(mode_ == NULL);
    }

    return ret;
}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

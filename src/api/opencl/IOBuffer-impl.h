#ifndef GMAC_API_OPENCL_IOBUFFER_IMPL_H_
#define GMAC_API_OPENCL_IOBUFFER_IMPL_H_

#include "Tracer.h"

namespace __impl { namespace opencl {

inline
IOBuffer::IOBuffer(void *addr, size_t size, bool async) :
    gmac::core::IOBuffer(addr, size, async), mode_(NULL), started_(false)
{
}

inline void
IOBuffer::toHost(Mode &mode)
{
    ASSERTION(started_ == false);
    state_  = ToHost;
    TRACE(LOCAL,"Buffer %p goes toHost", this); 
    mode_   = &mode;
}

inline void
IOBuffer::toAccelerator(Mode &mode)
{
    ASSERTION(started_ == false);
    state_  = ToAccelerator;
    TRACE(LOCAL,"Buffer %p goes toAccelerator", this);
    mode_   = &mode;
}

inline void
IOBuffer::started(cl_event event)
{
    ASSERTION(started_ == false);

    event_ = event;
    started_ = true;
}

inline gmacError_t
IOBuffer::wait()
{
    ASSERTION(state_ == Idle || started_ == true);

    gmacError_t ret = gmacSuccess;

    if (state_ != Idle) {
        ASSERTION(mode_ != NULL);
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

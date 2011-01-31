#ifndef GMAC_API_OPENCL_IOBUFFER_IMPL_H_
#define GMAC_API_OPENCL_IOBUFFER_IMPL_H_

#include "Tracer.h"

namespace __impl { namespace opencl {

inline
IOBuffer::IOBuffer(void *addr, size_t size) :
    gmac::core::IOBuffer(addr, size), mode_(NULL), started_(false)
{
}

inline void
IOBuffer::toHost(Mode &mode, cl_command_queue stream)
{
    ASSERTION(started_ == false);
    cl_int ret = clEnqueueMarker(stream, &start_);
    ASSERTION(ret == CL_SUCCESS);
    state_  = ToHost;
    TRACE(LOCAL,"Buffer %p goes toHost", this); 
    mode_   = &mode;
}

inline void
IOBuffer::toAccelerator(Mode &mode, cl_command_queue stream)
{
    ASSERTION(started_ == false);
    cl_int ret = clEnqueueMarker(stream, &start_);
    ASSERTION(ret == CL_SUCCESS);
    state_  = ToAccelerator;
    TRACE(LOCAL,"Buffer %p goes toAccelerator", this);
    mode_   = &mode;
}

inline void
IOBuffer::started(cl_event event)
{
    ASSERTION(started_ == false);

    end_ = event;
    started_ = true;
}

inline gmacError_t
IOBuffer::wait()
{
    ASSERTION(state_ == Idle || started_ == true);

    gmacError_t ret = gmacSuccess;

    if (state_ != Idle) {
        ASSERTION(mode_ != NULL);
        ret = mode_->waitForEvent(end_);
        ASSERTION(ret == gmacSuccess);
        if(state_ == ToHost) DataCommToHost(*mode_, start_, end_, size_);
        else if(state_ == ToAccelerator) DataCommToAccelerator(*mode_, start_, end_, size_);
        cl_int clret = clReleaseEvent(end_);
        ASSERTION(clret == CL_SUCCESS);
        clret = clReleaseEvent(start_);
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

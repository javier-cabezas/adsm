#ifndef GMAC_API_OPENCL_IOBUFFER_IMPL_H_
#define GMAC_API_OPENCL_IOBUFFER_IMPL_H_

namespace __impl { namespace opencl {

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
    ASSERTION(created_ == false);
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
        cl_int clret = clReleaseEvent(end_);
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

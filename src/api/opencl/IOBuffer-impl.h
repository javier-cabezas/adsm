#ifndef GMAC_API_OPENCL_IOBUFFER_IMPL_H_
#define GMAC_API_OPENCL_IOBUFFER_IMPL_H_

namespace __impl { namespace opencl {

inline void
IOBuffer::toHost(Mode &mode, cl_command_queue s)
{
    cl_int ret = CL_SUCCESS;
    cl_context ctx;
    ASSERTION(clGetCommandQueueInfo(s, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, NULL) == CL_SUCCESS);
    if (!created_) {
        end_ = clCreateUserEvent(ctx, &ret);
        created_ = true;
    }

    state_  = ToHost;
    TRACE(LOCAL,"Buffer %p goes toHost", this); 
    stream_ = s;
    mode_   = &mode;
}

inline void
IOBuffer::toAccelerator(Mode &mode, cl_command_queue s)
{
    if (!created_) {
        created_ = true;
    }

    state_  = ToAccelerator;
    TRACE(LOCAL,"Buffer %p goes toAccelerator", this);
    stream_ = s;
    mode_   = &mode;
}

inline void
IOBuffer::started()
{
    ASSERTION(created_ == true);
    cl_int ret = clEnqueueMarker(stream_, &end_);
    ASSERTION(ret == CL_SUCCESS);
}

inline gmacError_t
IOBuffer::wait()
{
    ASSERTION(state_ == Idle || created_ == true);

    gmacError_t ret = gmacSuccess;

    if (state_ != Idle) {
        ASSERTION(mode_ != NULL);
        ret = mode_->waitForEvent(end_);
        TRACE(LOCAL,"Buffer %p goes Idle", this);
        state_ = Idle;
        mode_  = NULL;
    } else {
        ASSERTION(mode_ == NULL);
    }

    return ret;
}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

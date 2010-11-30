#ifndef GMAC_API_CUDA_IOBUFFER_IMPL_H_
#define GMAC_API_CUDA_IOBUFFER_IMPL_H_

namespace __impl { namespace cuda {

inline void
IOBuffer::toHost(Mode &mode, CUstream s)
{
    if (!created_) {
        cuEventCreate(&start_, CU_EVENT_DEFAULT);
        cuEventCreate(&end_, CU_EVENT_DEFAULT);
        created_ = true;
    }

    cuEventRecord(start_, s);
    state_  = ToHost;
    TRACE(LOCAL,"Buffer %p goes toHost", this); 
    stream_ = s;
    mode_   = &mode;
}

inline void
IOBuffer::toAccelerator(Mode &mode, CUstream s)
{
    if (!created_) {
        cuEventCreate(&start_, CU_EVENT_DEFAULT);
        cuEventCreate(&end_, CU_EVENT_DEFAULT);
        created_ = true;
    }
    cuEventRecord(start_, s);
    state_  = ToAccelerator;
    TRACE(LOCAL,"Buffer %p goes toAccelerator", this);
    stream_ = s;
    mode_   = &mode;
}

inline void
IOBuffer::started()
{
    ASSERTION(created_ == true);

    cuEventRecord(end_, stream_);
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

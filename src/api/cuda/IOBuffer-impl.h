#ifndef GMAC_API_CUDA_IOBUFFER_IMPL_H_
#define GMAC_API_CUDA_IOBUFFER_IMPL_H_

#include "Tracer.h"

namespace __impl { namespace cuda {

inline void
IOBuffer::toHost(Mode &mode, CUstream s)
{
    if(created_ == false) {
        ASSERTION(cuEventCreate(&start_, CU_EVENT_DEFAULT) == CUDA_SUCCESS);
        ASSERTION(cuEventCreate(&end_, CU_EVENT_DEFAULT) == CUDA_SUCCESS);
        created_ = true;
    }

    ASSERTION(cuEventRecord(start_, s) == CUDA_SUCCESS);
    state_  = ToHost;
    TRACE(LOCAL,"Buffer %p goes toHost", this); 
    stream_ = s;
    mode_   = &mode;
}

inline void
IOBuffer::toAccelerator(Mode &mode, CUstream s)
{
    if(created_ == false) {
        ASSERTION(cuEventCreate(&start_, CU_EVENT_DEFAULT) == CUDA_SUCCESS);
        ASSERTION(cuEventCreate(&end_, CU_EVENT_DEFAULT) == CUDA_SUCCESS);
        created_ = true;
    }
    ASSERTION(cuEventRecord(start_, s) == CUDA_SUCCESS);
    state_  = ToAccelerator;
    TRACE(LOCAL,"Buffer %p goes toAccelerator", this);
    stream_ = s;
    mode_   = &mode;
}

inline void
IOBuffer::started()
{
    ASSERTION(created_ == true);
    ASSERTION(cuEventRecord(end_, stream_) == CUDA_SUCCESS);
}

inline gmacError_t
IOBuffer::wait()
{
    ASSERTION(state_ == Idle || created_ == true);

    gmacError_t ret = gmacSuccess;

    if (state_ != Idle) {
        ASSERTION(mode_ != NULL);
        ret = mode_->waitForEvent(end_);
        if(state_ == ToHost) DataCommToHost(*mode_, start_, end_, size_);
        else if(state_ == ToAccelerator) DataCommToAccelerator(*mode_, start_, end_, size_);
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

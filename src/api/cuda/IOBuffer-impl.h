#ifndef GMAC_API_CUDA_IOBUFFER_IMPL_H_
#define GMAC_API_CUDA_IOBUFFER_IMPL_H_

#include "Tracer.h"

namespace __impl { namespace cuda {

inline void
IOBuffer::toHost(Mode &mode, CUstream s)
{
    CUresult ret = CUDA_SUCCESS;

    EventMap::iterator it;
    it = map_.find(&mode);
    if (it == map_.end()) {
        CUevent start;
        CUevent end;
        ret = cuEventCreate(&start, CU_EVENT_DEFAULT);
        ASSERTION(ret == CUDA_SUCCESS);
        ret = cuEventCreate(&end, CU_EVENT_DEFAULT);
        ASSERTION(ret == CUDA_SUCCESS);

        std::pair<EventMap::iterator, bool> ret =
            map_.insert(EventMap::value_type(&mode, std::pair<CUevent, CUevent>(start, end)));
        it = ret.first;
    }

    ret = cuEventRecord(it->second.first, s);
    ASSERTION(ret == CUDA_SUCCESS);
    state_  = ToHost;
    TRACE(LOCAL,"Buffer %p goes toHost", this); 
    stream_ = s;
    mode_   = &mode;
}

inline void
IOBuffer::toAccelerator(Mode &mode, CUstream s)
{
    CUresult ret = CUDA_SUCCESS;

    EventMap::iterator it;
    it = map_.find(&mode);
    if (it == map_.end()) {
        CUevent start;
        CUevent end;
        ret = cuEventCreate(&start, CU_EVENT_DEFAULT);
        ASSERTION(ret == CUDA_SUCCESS);
        ret = cuEventCreate(&end, CU_EVENT_DEFAULT);
        ASSERTION(ret == CUDA_SUCCESS);

        std::pair<EventMap::iterator, bool> ret =
            map_.insert(EventMap::value_type(&mode, std::pair<CUevent, CUevent>(start, end)));
        it = ret.first;
    }

    ret = cuEventRecord(it->second.first, s);
    ASSERTION(ret == CUDA_SUCCESS);
    state_  = ToAccelerator;
    TRACE(LOCAL,"Buffer %p goes toAccelerator", this);
    stream_ = s;
    mode_   = &mode;
}

inline void
IOBuffer::started()
{
    EventMap::iterator it;
    it = map_.find(mode_);
    ASSERTION(state_ != Idle && it != map_.end());
    CUresult ret = cuEventRecord(it->second.second, stream_);
    ASSERTION(ret == CUDA_SUCCESS);
}

inline gmacError_t
IOBuffer::wait()
{
    EventMap::iterator it;
    it = map_.find(mode_);
    ASSERTION(state_ == Idle || it != map_.end());

    gmacError_t ret = gmacSuccess;

    if (state_ != Idle) {
        ASSERTION(mode_ != NULL);
        CUevent start = it->second.first;
        CUevent end   = it->second.second;
        trace::SetThreadState(trace::Wait);
        ret = mode_->waitForEvent(end);
        trace::SetThreadState(trace::Running);
        if(state_ == ToHost) DataCommToHost(*mode_, start, end, size_);
        else if(state_ == ToAccelerator) DataCommToAccelerator(*mode_, start, end, size_);
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

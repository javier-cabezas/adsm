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

        std::pair<EventMap::iterator, bool> par =
            map_.insert(EventMap::value_type(&mode, std::pair<CUevent, CUevent>(start, end)));
        it = par.first;
    }

    ret = cuEventRecord(it->second.first, s);
    ASSERTION(ret == CUDA_SUCCESS);
    state_  = ToHost;
    TRACE(LOCAL,"Buffer %p goes toHost", this); 
    stream_ = s;
    mode_   = &mode;

#ifdef EXPERIMENTAL
    map();
#endif
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

#ifdef EXPERIMENTAL
    map();
#endif
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
    return wait(false);
}

inline gmacError_t
IOBuffer::waitFromCUDA()
{
    return wait(true);
}



}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

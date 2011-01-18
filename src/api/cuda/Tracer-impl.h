#ifndef GMAC_API_CUDA_TRACER_IMPL_H_
#define GMAC_API_CUDA_TRACER_IMPL_H_

namespace __impl { namespace cuda { 

inline void DataCommunication(Mode &mode, THREAD_T src, THREAD_T dst, CUevent start,
        CUevent end, size_t size)
{
#if defined(USE_TRACE)
    uint64_t delta = 0;
    ASSERTION(mode.eventTime(delta, start, end) == gmacSuccess);
    return trace::DataCommunication(src, dst, delta, size);
#endif
}

inline void DataCommToAccelerator(Mode &mode, CUevent start, CUevent end, size_t size)
{
#if defined(USE_TRACE)
    return DataCommunication(mode, util::GetThreadId(), mode.id(), start, end, size);
#endif
}

inline void DataCommToHost(Mode &mode, CUevent start, CUevent end, size_t size)
{
#if defined(USE_TRACE)
    return DataCommunication(mode, mode.id(), util::GetThreadId(), start, end, size);
#endif
}


}}

#endif

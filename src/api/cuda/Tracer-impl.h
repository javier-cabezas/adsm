#ifndef GMAC_API_CUDA_TRACER_IMPL_H_
#define GMAC_API_CUDA_TRACER_IMPL_H_

namespace __impl { namespace cuda { 

inline void DataCommunication(THREAD_T src, THREAD_T dst, CUevent start,
        CUevent end, size_t size)
{
#if defined(USE_TRACE)
    float delta = 0.0;
    cuEventElapsedTime(&delta, start, end);
    return DataCommunication(src, dst, uint64_t(1000 * delta), size);
#endif
}

inline void DataCommunication(THREAD_T tid, CUevent start, CUevent end, size_t size)
{
#if defined(USE_TRACE)
    return DataCommunication(util::GetThreadId(), tid, start, end, size);
#endif
}

}}

#endif

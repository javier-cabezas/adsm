#ifndef GMAC_API_OPENCL_TRACER_IMPL_H_
#define GMAC_API_OPENCL_TRACER_IMPL_H_

namespace __impl { namespace opencl { 

inline void DataCommunication(Mode &mode, THREAD_T src, THREAD_T dst, cl_event start,
        cl_event end, size_t size)
{
#if defined(USE_TRACE)
    uint64_t delta = 0;
    gmacError_t ret = mode.eventTime(delta, start, end);
    ASSERTION(ret == gmacSuccess);
    return trace::DataCommunication(src, dst, delta, size);
#endif
}

inline void DataCommToAccelerator(Mode &mode, cl_event start, cl_event end, size_t size)
{
#if defined(USE_TRACE)
    return DataCommunication(mode, trace::GetThreadId(), mode.id(), start, end, size);
#endif
}

inline void DataCommToHost(Mode &mode, cl_event start, cl_event end, size_t size)
{
#if defined(USE_TRACE)
    return DataCommunication(mode, mode.id(), trace::GetThreadId(), start, end, size);
#endif
}


}}

#endif

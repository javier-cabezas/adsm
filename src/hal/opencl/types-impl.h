#ifndef GMAC_HAL_OPENCL_TYPES_IMPL_H_
#define GMAC_HAL_OPENCL_TYPES_IMPL_H_

//#include "device.h"
#include "util/Logger.h"

namespace __impl { namespace hal { namespace opencl {

inline
aspace_t::aspace_t(cl_context ctx, device &dev) :
    hal::detail::aspace_t<device, backend_traits>(ctx, dev)
{
}

inline
device &
aspace_t::get_device()
{
    return reinterpret_cast<device &>(hal::detail::aspace_t<device, backend_traits>::get_device());
}

inline
stream_t::stream_t(cl_command_queue stream, aspace_t &aspace) :
    hal::detail::stream_t<device, backend_traits>(stream, aspace)
{
}

inline
aspace_t &
stream_t::get_address_space()
{
    return reinterpret_cast<aspace_t &>(hal::detail::stream_t<device, backend_traits>::get_address_space());
}

inline
cl_event &
_event_common_t::operator()()
{
    return event_;
}

inline
const cl_event &
_event_common_t::operator()() const
{
    return event_;
}

inline
event_t::event_t(stream_t &stream, gmacError_t err) :
    hal::detail::event_t<device, backend_traits>(stream, err)
{
}

inline
stream_t &
event_t::get_stream()
{
    return reinterpret_cast<stream_t &>(hal::detail::event_t<device, backend_traits>::get_stream());
}

inline
async_event_t::async_event_t(stream_t &stream, gmacError_t err) :
    hal::detail::async_event_t<device, backend_traits>(stream, err)
{
}

inline
gmacError_t
async_event_t::sync()
{
    cl_int ret = clWaitForEvents(1, &event_);
    if (ret == CL_SUCCESS) {
        ret = clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_QUEUED, sizeof(hal::time_t), &timeQueued_, NULL);
        ASSERTION(ret == CL_SUCCESS);
        ret = clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_SUBMIT, sizeof(hal::time_t), &timeSubmit_, NULL);
        ASSERTION(ret == CL_SUCCESS);
        ret = clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_START, sizeof(hal::time_t), &timeStart_, NULL);
        ASSERTION(ret == CL_SUCCESS);
        ret = clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_END, sizeof(hal::time_t), &timeEnd_, NULL);
        ASSERTION(ret == CL_SUCCESS);
    }
    return error(ret);
}

inline
stream_t &
async_event_t::get_stream()
{
    return reinterpret_cast<stream_t &>(hal::detail::event_t<device, backend_traits>::get_stream());
}

}}}

#endif /* TYPES_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

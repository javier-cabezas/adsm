#ifndef GMAC_HAL_CUDA_TYPES_IMPL_H_
#define GMAC_HAL_CUDA_TYPES_IMPL_H_

//#include "Device.h"

namespace __impl { namespace hal { namespace cuda {

inline
aspace_t::aspace_t(CUcontext ctx, Device &device) :
    hal::detail::aspace_t<Device, backend_traits>(ctx, device)
{
}

inline
Device &
aspace_t::getDevice()
{
    return reinterpret_cast<Device &>(hal::detail::aspace_t<Device, backend_traits>::getDevice());
}

inline
stream_t::stream_t(CUstream stream, aspace_t &aspace) :
    hal::detail::stream_t<Device, backend_traits>(stream, aspace)
{
}

inline
aspace_t &
stream_t::getPASpace()
{
    return reinterpret_cast<aspace_t &>(hal::detail::stream_t<Device, backend_traits>::getPASpace());
}

inline
void
_event_common_t::begin(stream_t &stream)
{
    cuEventRecord(eventStart_, stream());
}

inline
void
_event_common_t::end(stream_t &stream)
{
    cuEventRecord(eventEnd_, stream());
}

inline
event_t::event_t(stream_t &stream, gmacError_t err) :
    hal::detail::event_t<Device, backend_traits>(stream, err)
{
}

inline
stream_t &
event_t::getStream()
{
    return reinterpret_cast<stream_t &>(hal::detail::event_t<Device, backend_traits>::getStream());
}

inline
async_event_t::async_event_t(stream_t &stream, gmacError_t err) :
    hal::detail::async_event_t<Device, backend_traits>(stream, err)
{
}

inline
gmacError_t
async_event_t::sync()
{
    CUresult ret = cuEventSynchronize(eventEnd_);
    if (ret == CUDA_SUCCESS) {
        float mili;
        ret = cuEventElapsedTime(&mili, eventStart_, eventEnd_);
        if (ret == CUDA_SUCCESS) {
            end_ = start_ + long_t(mili * 1000.f);
        }
    }
    return error(ret);
}

inline
stream_t &
async_event_t::getStream()
{
    return reinterpret_cast<stream_t &>(hal::detail::event_t<Device, backend_traits>::getStream());
}

}}}

#endif /* TYPES_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

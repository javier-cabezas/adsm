#ifndef GMAC_HAL_CUDA_TYPES_H_
#define GMAC_HAL_CUDA_TYPES_H_

#include <cuda.h>

#include "hal/types-detail.h"

namespace __impl { namespace hal { namespace cuda {

class Device;

typedef hal::detail::backend_traits<CUcontext, CUstream, CUevent> backend_traits;

gmacError_t error(CUresult err);

class aspace_t :
    public hal::detail::aspace_t<Device, backend_traits > {
public:
    aspace_t(CUcontext ctx, Device &device);

    Device &getDevice();
};

class stream_t :
    public hal::detail::stream_t<Device, backend_traits > {
public:
    stream_t(CUstream stream, aspace_t &aspace);

    aspace_t &getPASpace();
    CUstream &operator()();
};

class _event_common_t {
protected:
    CUevent eventStart_;
    CUevent eventEnd_;

    void begin(stream_t &stream);
    void end(stream_t &stream);
};

class event_t :
    public hal::detail::event_t<Device, backend_traits >,
    public _event_common_t {
    friend class Device;
public:
    event_t(stream_t &stream, gmacError_t err = gmacSuccess);
    stream_t &getStream();
};

class async_event_t :
    public hal::detail::async_event_t<Device, backend_traits >,
    public _event_common_t {
    friend class Device;
public:
    async_event_t(stream_t &stream, gmacError_t err = gmacSuccess);
    gmacError_t sync();
    stream_t &getStream();
};


}}}

#include "types-impl.h"

#endif /* GMAC_HAL_CUDA_TYPES_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

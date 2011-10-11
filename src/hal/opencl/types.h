#ifndef GMAC_HAL_OPENCL_TYPES_H_
#define GMAC_HAL_OPENCL_TYPES_H_

#include <CL/cl.h>

#include "hal/types-detail.h"

namespace __impl { namespace hal { namespace opencl {

class device;

typedef hal::detail::backend_traits<cl_context, cl_command_queue, cl_event> backend_traits;

gmacError_t error(cl_int err);

class aspace_t :
    public hal::detail::aspace_t<device, backend_traits> {
public:
    aspace_t(cl_context ctx, device &device);

    device &get_device();
};

class stream_t :
    public hal::detail::stream_t<device, backend_traits> {
public:
    stream_t(cl_command_queue stream, aspace_t &aspace);

    aspace_t &get_address_space();
    cl_command_queue &operator()();
};

class _event_common_t {
protected:
    _event_common_t()
    {
    }

    cl_event event_;

public:
    cl_event &operator()();
    const cl_event &operator()() const;
};

class event_t :
    public hal::detail::event_t<device, backend_traits>,
    public _event_common_t {
    friend class device;
public:
    event_t(stream_t &stream, gmacError_t err = gmacSuccess);
    stream_t &get_stream();
};

class async_event_t :
    public hal::detail::async_event_t<device, backend_traits>,
    public _event_common_t {
    friend class device;
public:
    async_event_t(stream_t &stream, gmacError_t err = gmacSuccess);
    gmacError_t sync();
    stream_t &get_stream();
};


}}}

#include "types-impl.h"

#endif /* GMAC_HAL_OPENCL_TYPES_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

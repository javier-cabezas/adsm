#ifndef GMAC_HAL_OPENCL_STREAM_H_
#define GMAC_HAL_OPENCL_STREAM_H_

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "hal/types-detail.h"

namespace __impl { namespace hal { namespace opencl {

class GMAC_LOCAL stream_t :
    public hal::detail::stream_t<backend_traits, implementation_traits> {

    typedef hal::detail::stream_t<backend_traits, implementation_traits> Parent;

    event_t lastEvent_;

    stream_t() = delete;
    stream_t(const stream_t &stream) = delete;

    stream_t &operator=(const stream_t &stream) = delete;
public:
    stream_t(cl_command_queue stream, context_t &context);

    context_t &get_context();

    Parent::state query();
    gmacError_t sync();
};

}}}

#endif /* GMAC_HAL_OPENCL_STREAM_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

#ifndef GMAC_HAL_TYPES_H_
#define GMAC_HAL_TYPES_H_

#include "config/common.h"

#include "include/gmac/types.h"

#ifdef USE_CUDA
#include "cuda/types.h"
#include "cuda/device.h"
#include "cuda/coherence_domain.h"
#include "ptr.h"

namespace __impl { namespace hal {
    typedef hal::detail::platform platform;
    typedef hal::detail::device device;

    typedef hal::detail::coherence_domain coherence_domain;

    typedef hal::detail::kernel kernel_t;
    typedef hal::cuda::kernel::config kernel_config;
    typedef hal::cuda::kernel::launch kernel_launch;
    typedef hal::detail::code_repository code_repository;
    typedef hal::detail::aspace aspace;
    typedef hal::detail::stream stream;
    //typedef hal::cuda::event_ptr event_ptr;
    typedef hal::detail::_event::type event_type;
    typedef hal::detail::event_ptr event_ptr;
    typedef hal::cuda::list_event list_event;

    typedef hal::detail::buffer buffer_t;

    //typedef hal::cuda::ptr_t ptr;
    //typedef hal::cuda::ptr_const_t const_ptr;
    //typedef hal::detail::ptr ptr;
    //typedef hal::detail::const_ptr const_ptr;

    typedef hal::cuda::list_platform list_platform;
}}

#else
#include "opencl/types.h"
#include "opencl/device.h"
#include "opencl/coherence_domain.h"

namespace __impl { namespace hal {
    typedef hal::opencl::platform platform;
    typedef hal::opencl::device device;

    typedef hal::opencl::coherence_domain coherence_domain;

    typedef hal::opencl::kernel_t kernel_t;
    typedef hal::opencl::code_repository code_repository;
    typedef hal::opencl::aspace aspace;
    typedef hal::opencl::stream_t stream_t;
    typedef hal::opencl::event_ptr event_ptr;
    typedef hal::opencl::list_event list_event;

    typedef hal::opencl::buffer_t buffer_t;

    typedef hal::opencl::ptr_t ptr;
    typedef hal::opencl::const_ptr_t const_ptr;
}}

#endif

namespace __impl { namespace hal {

gmacError_t
init();

gmacError_t
fini();

list_platform
get_platforms();

event_ptr copy(ptr dst, const_ptr src, size_t count, stream &stream, list_event &dependencies, gmacError_t &err);
event_ptr copy(ptr dst, const_ptr src, size_t count, stream &stream, event_ptr event, gmacError_t &err);
event_ptr copy(ptr dst, const_ptr src, size_t count, stream &stream, gmacError_t &err);

event_ptr copy(ptr dst, device_input &input, size_t count, stream &stream, list_event &dependencies, gmacError_t &err);
event_ptr copy(ptr dst, device_input &input, size_t count, stream &stream, event_ptr event, gmacError_t &err);
event_ptr copy(ptr dst, device_input &input, size_t count, stream &stream, gmacError_t &err);

event_ptr copy(device_output &output, const_ptr src, size_t count, stream &stream, list_event &dependencies, gmacError_t &err);
event_ptr copy(device_output &output, const_ptr src, size_t count, stream &stream, event_ptr event, gmacError_t &err);
event_ptr copy(device_output &output, const_ptr src, size_t count, stream &stream, gmacError_t &err);

event_ptr copy_async(ptr dst, const_ptr src, size_t count, stream &stream, list_event &dependencies, gmacError_t &err);
event_ptr copy_async(ptr dst, const_ptr src, size_t count, stream &stream, event_ptr event, gmacError_t &err);
event_ptr copy_async(ptr dst, const_ptr src, size_t count, stream &stream, gmacError_t &err);

event_ptr copy_async(ptr dst, device_input &input, size_t count, stream &stream, list_event &dependencies, gmacError_t &err);
event_ptr copy_async(ptr dst, device_input &input, size_t count, stream &stream, event_ptr event, gmacError_t &err);
event_ptr copy_async(ptr dst, device_input &input, size_t count, stream &stream, gmacError_t &err);

event_ptr copy_async(device_output &output, const_ptr src, size_t count, stream &stream, list_event &dependencies, gmacError_t &err);
event_ptr copy_async(device_output &output, const_ptr src, size_t count, stream &stream, event_ptr event, gmacError_t &err);
event_ptr copy_async(device_output &output, const_ptr src, size_t count, stream &stream, gmacError_t &err);

event_ptr memset(ptr dst, int c, size_t count, stream &stream, list_event &dependencies, gmacError_t &err);
event_ptr memset(ptr dst, int c, size_t count, stream &stream, event_ptr event, gmacError_t &err);
event_ptr memset(ptr dst, int c, size_t count, stream &stream, gmacError_t &err);

event_ptr memset_async(ptr dst, int c, size_t count, stream &stream, list_event &dependencies, gmacError_t &err);
event_ptr memset_async(ptr dst, int c, size_t count, stream &stream, event_ptr event, gmacError_t &err);
event_ptr memset_async(ptr dst, int c, size_t count, stream &stream, gmacError_t &err);

}}

#endif /* TYPES_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

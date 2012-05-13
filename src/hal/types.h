#ifndef GMAC_HAL_TYPES_H_
#define GMAC_HAL_TYPES_H_

#include "config/common.h"

#include "include/gmac/types.h"

#include "cpu/types.h"

#include "error.h"

#ifdef USE_CUDA
#include "cuda/types.h"
#include "cuda/phys/processing_unit.h"
#include "cuda/phys/platform.h"
#include "detail/ptr.h"

namespace __impl { namespace hal {

namespace phys {
    typedef hal::detail::phys::aspace aspace;
    typedef hal::detail::phys::memory memory;
    typedef hal::detail::phys::platform platform;
    typedef hal::detail::phys::processing_unit processing_unit;

    typedef hal::cuda::phys::list_platform list_platform;
}

namespace virt {
    typedef hal::detail::virt::aspace aspace;
    typedef hal::detail::virt::object object;
    typedef hal::detail::virt::object_view object_view;
}

typedef hal::detail::_event::type event_type;
typedef hal::detail::event_ptr event_ptr;
typedef hal::cuda::list_event list_event;

namespace code {
    typedef hal::detail::code::repository repository;

    typedef hal::detail::code::kernel kernel_t;
    typedef hal::cuda::code::kernel::config kernel_config;
    typedef hal::cuda::code::kernel::launch kernel_launch;
}

}}

#else
#include "opencl/types.h"

namespace __impl { namespace hal {
    typedef hal::opencl::platform platform;
    typedef hal::opencl::processing_unit processing_unit;

    typedef hal::opencl::kernel_t kernel_t;
    typedef hal::opencl::code_repository code_repository;
    typedef hal::opencl::aspace aspace;
    //typedef hal::opencl::stream_t stream_t;
    typedef hal::opencl::event_ptr event_ptr;
    typedef hal::opencl::list_event list_event;

    typedef hal::opencl::buffer_t buffer_t;

    typedef hal::opencl::ptr_t ptr;
    typedef hal::opencl::const_ptr_t const_ptr;
}}

#endif

namespace __impl { namespace hal {

error
init();

error
fini();

namespace phys {
list_platform
get_platforms();
}

event_ptr copy(ptr dst, const_ptr src, size_t count, list_event &dependencies, error &err);
event_ptr copy(ptr dst, const_ptr src, size_t count, event_ptr event, error &err);
event_ptr copy(ptr dst, const_ptr src, size_t count, error &err);

event_ptr copy(ptr dst, device_input &input, size_t count, list_event &dependencies, error &err);
event_ptr copy(ptr dst, device_input &input, size_t count, event_ptr event, error &err);
event_ptr copy(ptr dst, device_input &input, size_t count, error &err);

event_ptr copy(device_output &output, const_ptr src, size_t count, list_event &dependencies, error &err);
event_ptr copy(device_output &output, const_ptr src, size_t count, event_ptr event, error &err);
event_ptr copy(device_output &output, const_ptr src, size_t count, error &err);

event_ptr copy_async(ptr dst, const_ptr src, size_t count, list_event &dependencies, error &err);
event_ptr copy_async(ptr dst, const_ptr src, size_t count, event_ptr event, error &err);
event_ptr copy_async(ptr dst, const_ptr src, size_t count, error &err);

event_ptr copy_async(ptr dst, device_input &input, size_t count, list_event &dependencies, error &err);
event_ptr copy_async(ptr dst, device_input &input, size_t count, event_ptr event, error &err);
event_ptr copy_async(ptr dst, device_input &input, size_t count, error &err);

event_ptr copy_async(device_output &output, const_ptr src, size_t count, list_event &dependencies, error &err);
event_ptr copy_async(device_output &output, const_ptr src, size_t count, event_ptr event, error &err);
event_ptr copy_async(device_output &output, const_ptr src, size_t count, error &err);

event_ptr memset(ptr dst, int c, size_t count, list_event &dependencies, error &err);
event_ptr memset(ptr dst, int c, size_t count, event_ptr event, error &err);
event_ptr memset(ptr dst, int c, size_t count, error &err);

event_ptr memset_async(ptr dst, int c, size_t count, list_event &dependencies, error &err);
event_ptr memset_async(ptr dst, int c, size_t count, event_ptr event, error &err);
event_ptr memset_async(ptr dst, int c, size_t count, error &err);

}}

#endif /* TYPES_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

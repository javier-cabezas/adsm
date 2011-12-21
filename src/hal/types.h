#ifndef GMAC_HAL_TYPES_H_
#define GMAC_HAL_TYPES_H_

#include "config/common.h"

#include "include/gmac/types.h"

#ifdef USE_CUDA
#include "cuda/types.h"
#include "cuda/device.h"
#include "cuda/coherence_domain.h"

namespace __impl { namespace hal {
    typedef hal::cuda::device device;

    typedef hal::cuda::coherence_domain coherence_domain;

    typedef hal::cuda::kernel_t kernel_t;
    typedef hal::cuda::code_repository code_repository;
    typedef hal::cuda::context_t context_t;
    typedef hal::cuda::stream_t stream_t;
    typedef hal::cuda::event_ptr event_ptr;
    typedef hal::cuda::list_event list_event;

    typedef hal::cuda::buffer_t buffer_t;

    typedef hal::cuda::ptr_t ptr;
    typedef hal::cuda::ptr_const_t const_ptr;
}}

#else
#include "opencl/types.h"
#include "opencl/device.h"
#include "opencl/coherence_domain.h"

namespace __impl { namespace hal {
    typedef hal::opencl::device device;

    typedef hal::opencl::coherence_domain coherence_domain;

    typedef hal::opencl::kernel_t kernel_t;
    typedef hal::opencl::code_repository code_repository;
    typedef hal::opencl::context_t context_t;
    typedef hal::opencl::stream_t stream_t;
    typedef hal::opencl::event_ptr event_ptr;
    typedef hal::opencl::list_event list_event;

    typedef hal::opencl::buffer_t buffer_t;

    typedef hal::opencl::ptr_t ptr;
    typedef hal::opencl::ptr_const_t const_ptr;
}}

#endif

namespace __impl { namespace hal {

gmacError_t
init_platform();

std::list<device *>
init_devices();

}}

#endif /* TYPES_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

#ifndef GMAC_HAL_TYPES_H_
#define GMAC_HAL_TYPES_H_

#include "config/common.h"

#include "include/gmac/types.h"

#ifdef USE_CUDA
#include "cuda/types.h"
#include "cuda/Device.h"
#include "cuda/CoherenceDomain.h"

namespace __impl { namespace hal {
typedef hal::cuda::Device Device;

typedef hal::cuda::CoherenceDomain CoherenceDomain;

typedef hal::cuda::aspace_t aspace_t;
typedef hal::cuda::stream_t stream_t;
typedef hal::cuda::event_t event_t;
typedef hal::cuda::async_event_t async_event_t;
}}
#else
#include "opencl/types.h"
namespace __impl { namespace hal {
typedef hal::opencl::Device Device;

typedef hal::opencl::aspace_t aspace_t;
typedef hal::opencl::stream_t stream_t;
typedef hal::opencl::event_t event_t;
typedef hal::cuda::async_event_t async_event_t;
}}
#endif

#endif /* TYPES_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

#ifndef GMAC_HAL_CUDA_TYPES_H_
#define GMAC_HAL_CUDA_TYPES_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "hal/types-detail.h"

#include "ptr.h"

namespace __impl { namespace hal { namespace cuda {

class coherence_domain;
class platform;
class device;
class aspace;
class stream;
class kernel;
class texture_t;
class variable_t;
class code_repository;
class _event_t;
typedef util::shared_ptr<_event_t> event_ptr;
class event_list;
class buffer_t;

typedef __impl::hal::_ptr_t<
                             _cuda_ptr_t,
                             aspace,
                             device
                           > ptr_t;

typedef __impl::hal::_const_ptr_t<
                                   _cuda_ptr_t,
                                   aspace,
                                   device
                                 > ptr_const_t;

typedef hal::detail::implementation_traits<coherence_domain,
                                           platform,
                                           device,
                                           aspace,
                                           stream,
                                           kernel,
                                           texture_t,
                                           variable_t,
                                           code_repository,
                                           _event_t,
                                           event_ptr,
                                           event_list,
                                           buffer_t,
                                           ptr_t,
                                           ptr_const_t> implementation_traits;

gmacError_t error(CUresult err);


typedef std::list<platform *> list_platform;

}

cuda::list_platform get_platforms();

}}

#include "event.h"
#include "aspace.h"
#include "kernel.h"
#include "stream.h"

#include "aspace-impl.h"
#include "stream-impl.h"
#include "kernel-impl.h"
#include "event-impl.h"

#include "module.h"

#endif /* GMAC_HAL_CUDA_TYPES_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

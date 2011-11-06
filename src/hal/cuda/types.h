#ifndef GMAC_HAL_CUDA_TYPES_H_
#define GMAC_HAL_CUDA_TYPES_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "hal/types-detail.h"

#include "ptr.h"

namespace __impl { namespace hal { namespace cuda {

class device;

class coherence_domain;
class context_t;
class stream_t;
class kernel_t;
class texture_t;
class variable_t;
class code_repository;
class _event_t;
class event_t;
class buffer_t;

typedef __impl::hal::_ptr_t<_cuda_ptr_t,
                            context_t> ptr_t;

typedef hal::detail::backend_traits<CUcontext,
                                    CUstream,
                                    CUevent,
                                    CUfunction,
                                    dim3> backend_traits;

typedef hal::detail::implementation_traits<coherence_domain,
                                           context_t,
                                           stream_t,
                                           kernel_t,
                                           texture_t,
                                           variable_t,
                                           code_repository,
                                           event_t,
                                           buffer_t,
                                           ptr_t> implementation_traits;

gmacError_t error(CUresult err);

}}}

#include "event.h"
#include "context.h"
#include "kernel.h"
#include "stream.h"

#include "context-impl.h"
#include "stream-impl.h"
#include "kernel-impl.h"
#include "event-impl.h"

#include "module.h"

#endif /* GMAC_HAL_CUDA_TYPES_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

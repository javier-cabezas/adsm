#ifndef GMAC_HAL_CUDA_TYPES_H_
#define GMAC_HAL_CUDA_TYPES_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "hal/detail/types.h"

#include "ptr.h"

namespace __impl { namespace hal {

namespace cuda {

namespace phys {
    class coherence_domain;
    class processing_unit;
    class platform;
    class aspace;
    typedef hal::detail::phys::memory     memory;
}

namespace virt {
    class aspace;
}

namespace code {
class kernel;
class repository;
}

class stream;
class texture_t;
class variable_t;
class _event_t;
typedef util::shared_ptr<_event_t> event_ptr;
class event_list;
class buffer_t;

#if 0
typedef __impl::hal::_ptr_t<
                             _cuda_ptr_t,
                             virt::aspace,
                             phys::processing_unit
                           > ptr_t;

typedef __impl::hal::_const_ptr_t<
                                   _cuda_ptr_t,
                                   virt::aspace,
                                   phys::processing_unit
                                 > ptr_const_t;

typedef __impl::hal::ptr_t ptr_t;
#endif

gmacError_t error(CUresult err);

namespace phys {
    typedef std::list<platform *> list_platform;
}

}

namespace phys {
    cuda::phys::list_platform get_platforms();
}

}}

#include "event.h"
#include "phys/aspace.h"
#include "virt/aspace.h"
#include "kernel.h"
#include "stream.h"

#include "virt/aspace-impl.h"
#include "stream-impl.h"
#include "kernel-impl.h"
#include "event-impl.h"

#include "module.h"

#endif /* GMAC_HAL_CUDA_TYPES_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

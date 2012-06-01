#ifndef GMAC_HAL_CPU_TYPES_H_
#define GMAC_HAL_CPU_TYPES_H_

#include "hal/detail/types.h"

namespace __impl { namespace hal {

namespace cpu {

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
    class repository_view;
}

class stream;
class texture_t;
class variable_t;
class event_list;
class buffer_t;

}

}}

#include "operation.h"
#include "phys/aspace.h"
#include "phys/processing_unit.h"
#include "virt/aspace.h"
#include "virt/scheduler.h"

#include "operation-impl.h"

#endif /* GMAC_HAL_CPU_TYPES_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

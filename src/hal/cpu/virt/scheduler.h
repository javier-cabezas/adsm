#ifndef GMAC_HAL_CPU_VIRT_SCHEDULER_H_
#define GMAC_HAL_CPU_VIRT_SCHEDULER_H_

#include "hal/cpu/types.h"

namespace __impl { namespace hal { namespace cpu { namespace virt {

typedef hal::detail::virt::context hal_context;
typedef hal::detail::virt::scheduler hal_scheduler;

class GMAC_LOCAL scheduler :
    public hal_scheduler
{
public:
    hal::error add_context(hal_context &ctx)
    {
        FATAL("Scheduling not implemented in CPU");
        return hal::error::HAL_SUCCESS;
    }
};

}}}}

#endif /* ASPACE_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

#ifndef GMAC_HAL_DETAIL_PHYS_MEMORY_IMPL_H_
#define GMAC_HAL_DETAIL_PHYS_MEMORY_IMPL_H_

#include "processing_unit.h"

namespace __impl { namespace hal { namespace detail { namespace phys {

#if 0
bool
memory::is_host_memory() const
{
    for (auto p : attachedUnits_) {
        if (p->get_type() == processing_unit::PUNIT_TYPE_CPU) {
            return true;
        }
    }

    return false;
}
#endif

}}}}

#endif


/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

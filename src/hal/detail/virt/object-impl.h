#ifndef GMAC_HAL_DETAIL_PHYS_OBJECT_IMPL_H_
#define GMAC_HAL_DETAIL_PHYS_OBJECT_IMPL_H_

#include "hal/detail/types.h"

namespace __impl { namespace hal { namespace detail {
    
namespace virt {

inline
const phys::memory &
object::get_memory() const
{
    return *memory_;
}

}

}}}

#endif /* OBJECT_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

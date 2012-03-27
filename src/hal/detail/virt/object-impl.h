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

inline
object::set_view
object::get_views() const
{
    set_view ret;

    for (auto v : views_) {
        ret.insert(v.second);
    }

    return ret;
}

inline
object::set_view
object::get_views(detail::phys::processing_unit::type type) const
{
    set_view ret;

    for (auto v : views_) {
        const aspace::set_processing_unit &pus = v.first->get_processing_units();

        if (util::algo::has_predicate(pus, [type](const detail::phys::processing_unit *pu) -> bool
                                           {
                                               return pu->get_type() == type;
                                           })) 
        {
            ret.insert(v.second);
        }
    }

    return ret;
}

inline
object::set_view
object::get_views(detail::phys::aspace &as) const
{
    set_view ret;

    for (auto v : views_) {
        if (&v.first->get_paspace() == &as)
        {
            ret.insert(v.second);
        }
    }

    return ret;
}

}

}}}

#endif /* OBJECT_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

#ifndef GMAC_HAL_DETAIL_PHYS_PLATFORM_IMPL_H_
#define GMAC_HAL_DETAIL_PHYS_PLATFORM_IMPL_H_

#include "aspace.h"

namespace __impl { namespace hal { namespace detail { namespace phys {

inline
platform::~platform()
{
    for (auto pUnit : pUnits_) {
        delete pUnit;
    }

    pUnits_.clear();
}

inline
void
platform::add_processing_unit(processing_unit &pUnit)
{
    pUnits_.insert(&pUnit);
}

inline
void
platform::add_paspace(aspace_ptr as)
{
    aspaces_.insert(as);
    memories_.insert(as->get_memories().begin(),
                     as->get_memories().end());
}

inline
const platform::set_processing_unit &
platform::get_processing_units() const
{
    return pUnits_;
}

inline
const platform::set_memory &
platform::get_memories() const
{
    return memories_;
}

inline
const platform::set_aspace &
platform::get_paspaces() const
{
    return aspaces_;
}

}}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

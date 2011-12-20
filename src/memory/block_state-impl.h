#ifndef GMAC_MEMORY_BLOCK_STATE_IMPL_H_
#define GMAC_MEMORY_BLOCK_STATE_IMPL_H_

#include <algorithm>

#include "core/address_space.h"

namespace __impl { namespace memory {

#if 0
inline
template<typename State>
block_state::block_state(object_state<State> &parent,
                         hostptr_t hostAddr,
                         hostptr_t shadowAddr,
                         size_t size,
                         lazy_types::State init) :
    memory::protocols::common::block_state(hostAddr, shadowAddr, size),
    state_(init),
    parent_(parent)
{
}

template<typename State>
inline core::address_space_ptr
block_state<State>::get_owner() const
{
    return parent_.get_owner();
}

template<typename State>
inline accptr_t
block_state<State>::get_device_addr(const hostptr_t addr) const
{
    return parent_.get_device_addr(addr);
}

template<typename State>
inline accptr_t
block_state<State>::get_device_addr() const
{
    return get_device_addr(this->addr_);
}
#endif
}}

#endif

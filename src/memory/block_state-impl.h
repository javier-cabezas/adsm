#ifndef GMAC_MEMORY_BLOCK_STATE_IMPL_H_
#define GMAC_MEMORY_BLOCK_STATE_IMPL_H_

#include <algorithm>

#include "core/address_space.h"

namespace __impl { namespace memory {

template<typename State>
inline
block_state<State>::block_state(object_state<State> &parent,
                                hostptr_t hostAddr,
                                hostptr_t shadowAddr,
                                size_t size,
                                typename State::protocol_state init) :
    gmac::memory::block(hostAddr, shadowAddr, size),
    State(init),
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

}}

#endif

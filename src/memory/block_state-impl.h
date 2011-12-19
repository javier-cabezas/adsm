#ifndef GMAC_MEMORY_GENERICBLOCK_INST_H_
#define GMAC_MEMORY_GENERICBLOCK_INST_H_

#include <algorithm>

#include "core/address_space.h"

namespace __impl { namespace memory {

template<typename State>
inline
block_state<State>::block_state(object_state<State> &parent,
                                  hostptr_t hostAddr,
                                  hostptr_t shadowAddr,
                                  size_t size,
                                  typename State::ProtocolState init) :
    gmac::memory::block(hostAddr, shadowAddr, size),
    State(init),
    parent_(parent)
{
}

template<typename State>
inline core::address_space_ptr
block_state<State>::owner() const
{
    return parent_.owner();
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

template<typename State>
inline hal::event_ptr
block_state<State>::to_host(unsigned blockOff, size_t count, gmacError_t &err)
{
    hal::event_ptr ret;

    err = parent_.owner()->copy(hal::ptr_t(this->shadow_ + blockOff),
                                get_device_addr() + blockOff,
                                count);

    return ret;
}

template<typename State>
inline hal::event_ptr
block_state<State>::to_accelerator(unsigned blockOff, size_t count, gmacError_t &err)
{
    hal::event_ptr ret;
    ret = parent_.owner()->copy_async(get_device_addr() + blockOff,
                                      hal::ptr_t(shadow_ + blockOff),
                                      count, err);

    return ret;
}

}}

#endif

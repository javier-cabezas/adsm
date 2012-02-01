#ifdef USE_DBC

#ifndef GMAC_MEMORY_DBC_OBJECT_STATE_IMPL_H_
#define GMAC_MEMORY_DBC_OBJECT_STATE_IMPL_H_

#include "memory/object_state.h"

namespace __dbc { namespace memory {

template <typename ProtocolTraits>
object_state<ProtocolTraits>::object_state(protocol_impl &protocol, host_ptr addr, size_t size,
                                           int flagsHost, int flagsDevice,
                                           typename ProtocolTraits::State init, gmacError_t &err) :
    parent(protocol, addr, size, flagsHost, flagsDevice, init, err)
{
    REQUIRES(size > 0);

    ENSURES(parent::addr_ || err != gmacSuccess);
}

template <typename ProtocolTraits>
object_state<ProtocolTraits>::~object_state()
{
}

template <typename ProtocolTraits>
typename object_state<ProtocolTraits>::ptr_impl
object_state<ProtocolTraits>::get_device_addr(host_ptr addr)
{
    REQUIRES(addr >= this->get_bounds().start);
    REQUIRES(addr <  this->get_bounds().end);

    ptr_impl ret = parent::get_device_addr(addr);

    return ret;
}

template <typename ProtocolTraits>
typename object_state<ProtocolTraits>::const_ptr_impl
object_state<ProtocolTraits>::get_device_const_addr(host_const_ptr addr) const
{
    REQUIRES(addr >= this->get_bounds().start);
    REQUIRES(addr <  this->get_bounds().end);

    const_ptr_impl ret = parent::get_device_const_addr(addr);

    return ret;
}

}}

#endif
#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

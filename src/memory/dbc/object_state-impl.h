#ifdef USE_DBC

#ifndef GMAC_MEMORY_DBC_OBJECT_STATE_IMPL_H_
#define GMAC_MEMORY_DBC_OBJECT_STATE_IMPL_H_

#include "memory/object_state.h"

namespace __dbc { namespace memory {

template <typename ProtocolTraits>
object_state<ProtocolTraits>::object_state(protocol_impl &protocol, host_ptr addr, size_t size, typename ProtocolTraits::State init, gmacError_t &err) :
    parent(protocol, addr, size, init, err)
{
    REQUIRES(size > 0);

    ENSURES(parent::addr_ || err != gmacSuccess);
}

template <typename ProtocolTraits>
object_state<ProtocolTraits>::~object_state()
{
}

}}

#endif
#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

#if 0
#ifdef USE_DBC

#include "memory/block.h"

namespace __dbc { namespace memory {

block::block(host_ptr addr, host_ptr shadow, size_t size) :
    __impl::memory::block(addr, shadow, size)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    REQUIRES(addr != NULL);
    REQUIRES(shadow != NULL);
}

}}

#endif
#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

#ifdef USE_DBC

#include "memory/block.h"

namespace __dbc { namespace memory {

block::block(hostptr_t addr, hostptr_t shadow, size_t size) :
    __impl::memory::block(addr, shadow, size)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    REQUIRES(addr != NULL);
    REQUIRES(shadow != NULL);
}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

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

block::~block()
{
}

#if 0
gmacError_t
block::memoryOp(__impl::memory::protocol_interface::MemoryOp op, __impl::core::io_buffer &buffer, size_t size, size_t bufferOffset, size_t blockOffset)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    REQUIRES(blockOffset + size <= size_);
    REQUIRES(bufferOffset + size <= buffer.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::block::memoryOp(op, buffer, size, bufferOffset, blockOffset);
    // POSTCONDITIONS
    
    return ret;
}

gmacError_t
block::memset(int v, size_t size, size_t blockOffset)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    REQUIRES(blockOffset + size <= size_);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::block::memset(v, size, blockOffset);
    // POSTCONDITIONS
    
    return ret;
}
#endif


}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

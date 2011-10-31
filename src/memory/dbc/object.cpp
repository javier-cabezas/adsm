#ifdef USE_DBC

#include "core/io_buffer.h"
#include "memory/object.h"

namespace __dbc { namespace memory {

object::object(hostptr_t addr, size_t size) :
    __impl::memory::object(addr, size)
{
}

object::~object()
{
}

gmacError_t
object::memoryOp(__impl::memory::Protocol::MemoryOp op, __impl::core::io_buffer &buffer, size_t size, size_t bufferOffset, size_t objectOffset)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    REQUIRES(objectOffset + size <= size_);
    REQUIRES(bufferOffset + size <= buffer.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::object::memoryOp(op, buffer, size, bufferOffset, objectOffset);
    // POSTCONDITIONS
    
    return ret;

}


ssize_t
object::blockBase(size_t offset) const
{
    // PRECONDITIONS
    REQUIRES(offset <= size_);
    // CALL IMPLEMENTATION
    ssize_t ret = __impl::memory::object::blockBase(offset);
    // POSTCONDITIONS
    
    return ret;
}

size_t
object::blockEnd(size_t offset) const
{
    // PRECONDITIONS
    REQUIRES(offset <= size_);
    // CALL IMPLEMENTATION
    size_t ret = __impl::memory::object::blockEnd(offset);
    // POSTCONDITIONS
    
    return ret;
}

gmacError_t
object::signalRead(hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr >= addr_);
    REQUIRES(addr  < addr_ + size_);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::object::signalRead(addr);
    // POSTCONDITIONS
    
    return ret;
}

gmacError_t
object::signalWrite(hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr >= addr_);
    REQUIRES(addr  < addr_ + size_);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::object::signalWrite(addr);
    // POSTCONDITIONS
    
    return ret;
}

gmacError_t
object::copyToBuffer(__impl::core::io_buffer &buffer, size_t size, 
        size_t bufferOffset, size_t objectOffset)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    REQUIRES(objectOffset + size <= size_);
    REQUIRES(bufferOffset + size <= buffer.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::object::copyToBuffer(buffer, size, bufferOffset, objectOffset);
    // POSTCONDITIONS
    
    return ret;
}

gmacError_t
object::copyFromBuffer(__impl::core::io_buffer &buffer, size_t size, 
        size_t bufferOffset, size_t objectOffset)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    REQUIRES(objectOffset + size <= size_);
    REQUIRES(bufferOffset + size <= buffer.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::object::copyFromBuffer(buffer, size, bufferOffset, objectOffset);
    // POSTCONDITIONS
    
    return ret;
}

gmacError_t
object::memset(size_t offset, int v, size_t size)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    REQUIRES(offset + size <= size_);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::object::memset(offset, v, size);
    // POSTCONDITIONS
    
    return ret;

}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

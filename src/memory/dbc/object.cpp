#ifdef USE_DBC

#include "memory/object.h"

namespace __dbc { namespace memory {

object::object(protocol_impl &protocol, host_ptr addr, size_t size) :
    parent(protocol, addr, size)
{
    REQUIRES(size > 0);
}

object::~object()
{
}

ssize_t
object::get_block_base(size_t offset) const
{
    // PRECONDITIONS
    REQUIRES(offset <= size_);
    // CALL IMPLEMENTATION
    ssize_t ret = parent::get_block_base(offset);
    // POSTCONDITIONS
    
    return ret;
}

size_t
object::get_block_end(size_t offset) const
{
    // PRECONDITIONS
    REQUIRES(offset <= size_);
    // CALL IMPLEMENTATION
    size_t ret = parent::get_block_end(offset);
    // POSTCONDITIONS
    
    return ret;
}

object::event_ptr_impl
object::signal_read(host_ptr addr, gmacError_t &err)
{
    // PRECONDITIONS
    REQUIRES(addr >= addr_);
    REQUIRES(addr  < addr_ + size_);
    // CALL IMPLEMENTATION
    event_ptr_impl ret = parent::signal_read(addr, err);
    // POSTCONDITIONS
    
    return ret;
}

object::event_ptr_impl
object::signal_write(host_ptr addr, gmacError_t &err)
{
    // PRECONDITIONS
    REQUIRES(addr >= addr_);
    REQUIRES(addr  < addr_ + size_);
    // CALL IMPLEMENTATION
    event_ptr_impl ret = parent::signal_write(addr, err);
    // POSTCONDITIONS
    
    return ret;
}

gmacError_t
object::to_io_device(device_output_impl &output, size_t offset, size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(offset + count <= size_);
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::to_io_device(output, offset, count);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
object::from_io_device(size_t offset, device_input_impl &input, size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(offset + count <= size_);
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::from_io_device(offset, input, count);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
object::memset(size_t offset, int v, size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(offset + count <= size_);
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::memset(offset, v, count);
    // POSTCONDITIONS
    
    return ret;

}

gmacError_t
object::memcpy_to_object(size_t offset,
                         host_const_ptr src, size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(offset + count <= size_);
    REQUIRES(src);
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::memcpy_to_object(offset, src, count);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
object::memcpy_object_to_object(object &dstObj, size_t dstOffset,
                                size_t srcOffset,
                                size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(srcOffset + count <= size_);
    REQUIRES(dstOffset + count <= dstObj.size_);
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::memcpy_object_to_object(dstObj, dstOffset, srcOffset, count);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
object::memcpy_from_object(host_ptr dst,
                           size_t offset, size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(offset + count <= size_);
    REQUIRES(dst);
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::memcpy_from_object(dst, offset, count);
    // POSTCONDITIONS

    return ret;
}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

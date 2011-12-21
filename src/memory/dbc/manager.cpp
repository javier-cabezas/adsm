#ifdef USE_DBC

#include "manager.h"

namespace __dbc { namespace memory {

manager::manager(process_impl &proc) :
    parent(proc)
{
}

manager::~manager()
{
}

#if 0
gmacError_t
manager::map(void *addr, size_t size, GmacProtection prot)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::map(addr, size, prot);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
manager::unmap(void *addr, size_t size)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::unmap(addr, size);
    // POSTCONDITIONS

    return ret;
}
#endif

size_t
manager::get_alloc_size(address_space_ptr_impl aspace, host_const_ptr addr, gmacError_t &err) const
{
    // PRECONDITIONS
    REQUIRES(bool(aspace));
    REQUIRES(addr);
    // CALL IMPLEMENTATION
    size_t ret = parent::get_alloc_size(aspace, addr, err);
    // POSTCONDITIONS
    ENSURES(ret > 0 || err != gmacSuccess);

    return ret;
}

gmacError_t
manager::alloc(address_space_ptr_impl aspace, host_ptr *addr, size_t size)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::alloc(aspace, addr, size);
    // POSTCONDITIONS

    return ret;
}

#if 0
gmacError_t
manager::globalAlloc(void **addr, size_t size, GmacGlobalMallocType hint)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::globalAlloc(addr, size, hint);
    // POSTCONDITIONS

    return ret;
}
#endif

gmacError_t
manager::free(address_space_ptr_impl aspace, host_ptr addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::free(aspace, addr);
    // POSTCONDITIONS

    return ret;
}


manager::address_space_ptr_impl
manager::get_owner(host_const_ptr addr, size_t size)
{
    // PRECONDITIONS
    REQUIRES(bool(addr));
    // CALL IMPLEMENTATION
    address_space_ptr_impl ret = parent::get_owner(addr, size);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
manager::acquire_objects(address_space_ptr_impl aspace, const list_addr_impl &addrs)
{
    // PRECONDITIONS
    REQUIRES(bool(aspace));
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::acquire_objects(aspace, addrs);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
manager::release_objects(address_space_ptr_impl aspace, const list_addr_impl &addrs)
{
    // PRECONDITIONS
    REQUIRES(bool(aspace));
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::release_objects(aspace, addrs);
    // POSTCONDITIONS

    return ret;
}

bool
manager::signal_read(address_space_ptr_impl aspace, host_ptr addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    bool ret = parent::signal_read(aspace, addr);
    // POSTCONDITIONS

    return ret;
}

bool
manager::signal_write(address_space_ptr_impl aspace, host_ptr addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    bool ret = parent::signal_write(aspace, addr);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
manager::from_io_device(address_space_ptr_impl aspace, host_ptr addr, device_input_impl &input, size_t count)
{
    // PRECONDITIONS
    REQUIRES(bool(aspace));
    REQUIRES(addr != NULL);
    REQUIRES(count > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::from_io_device(aspace, addr, input, count);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
manager::to_io_device(device_output_impl &output, address_space_ptr_impl aspace, host_const_ptr addr, size_t count)
{
    // PRECONDITIONS
    REQUIRES(bool(aspace));
    REQUIRES(addr != NULL);
    REQUIRES(count > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::to_io_device(output, aspace, addr, count);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
manager::memcpy(address_space_ptr_impl aspace, host_ptr dst, host_const_ptr src, size_t n)
{
    // PRECONDITIONS
    REQUIRES(src != NULL);
    REQUIRES(dst != NULL);
    REQUIRES(n > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::memcpy(aspace, dst, src, n);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
manager::memset(address_space_ptr_impl aspace, host_ptr dst, int c, size_t n)
{
    // PRECONDITIONS
    REQUIRES(dst != NULL);
    REQUIRES(n > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::memset(aspace, dst, c, n);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
manager::flush_dirty(address_space_ptr_impl aspace)
{
    // PRECONDITIONS
    REQUIRES(bool(aspace));
    // CALL IMPLEMENTATION
    gmacError_t ret = parent::flush_dirty(aspace);
    // POSTCONDITIONS

    return ret;
}

}}

#endif

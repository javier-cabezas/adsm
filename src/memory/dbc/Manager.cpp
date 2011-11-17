#ifdef USE_DBC

#include "Manager.h"

namespace __dbc { namespace memory {

manager::manager(ProcessImpl &proc) :
    Parent(proc)
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
    gmacError_t ret = Parent::map(addr, size, prot);
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
    gmacError_t ret = Parent::unmap(addr, size);
    // POSTCONDITIONS

    return ret;
}
#endif
gmacError_t
manager::alloc(address_space_impl aspace, hostptr_t *addr, size_t size)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::alloc(aspace, addr, size);
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
    gmacError_t ret = Parent::globalAlloc(addr, size, hint);
    // POSTCONDITIONS

    return ret;
}
#endif

gmacError_t
manager::free(address_space_impl aspace, hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::free(aspace, addr);
    // POSTCONDITIONS

    return ret;
}

bool
manager::signal_read(address_space_impl aspace, hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    bool ret = Parent::signal_read(aspace, addr);
    // POSTCONDITIONS

    return ret;
}

bool
manager::signal_write(address_space_impl aspace, hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    bool ret = Parent::signal_write(aspace, addr);
    // POSTCONDITIONS

    return ret;
}

#if 0
gmacError_t
manager::toIOBuffer(address_space_impl aspace, io_buffer_impl &buffer, size_t bufferOff, const hostptr_t addr, size_t size)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    REQUIRES(size > 0);
    REQUIRES(size <= buffer.size() - bufferOff);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::toIOBuffer(aspace, buffer, bufferOff, addr, size);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
manager::fromIOBuffer(address_space_impl aspace, hostptr_t addr, io_buffer_impl &buffer, size_t bufferOff, size_t size)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    REQUIRES(size > 0);
    REQUIRES(size <= buffer.size() - bufferOff);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::fromIOBuffer(aspace, addr, buffer, bufferOff, size);
    // POSTCONDITIONS

    return ret;
}
#endif

gmacError_t
manager::memcpy(address_space_impl aspace, hostptr_t dst, const hostptr_t src, size_t n)
{
    // PRECONDITIONS
    REQUIRES(src != NULL);
    REQUIRES(dst != NULL);
    REQUIRES(n > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::memcpy(aspace, dst, src, n);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
manager::memset(address_space_impl aspace, hostptr_t dst, int c, size_t n)
{
    // PRECONDITIONS
    REQUIRES(dst != NULL);
    REQUIRES(n > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::memset(aspace, dst, c, n);
    // POSTCONDITIONS

    return ret;
}

}}

#endif

#ifdef USE_DBC

#include "core/io_buffer.h"

#include "Manager.h"

namespace __dbc { namespace memory {

Manager::Manager(ProcessImpl &proc) :
    Parent(proc)
{
}

Manager::~Manager()
{
}
#if 0
gmacError_t
Manager::map(void *addr, size_t size, GmacProtection prot)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::map(addr, size, prot);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Manager::unmap(void *addr, size_t size)
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
Manager::alloc(address_space_impl aspace, hostptr_t *addr, size_t size)
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
Manager::globalAlloc(void **addr, size_t size, GmacGlobalMallocType hint)
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
Manager::free(address_space_impl aspace, hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    gmacError_t ret = Parent::free(aspace, addr);
    // POSTCONDITIONS

    return ret;
}

bool
Manager::signalRead(address_space_impl aspace, hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    bool ret = Parent::signalRead(aspace, addr);
    // POSTCONDITIONS

    return ret;
}

bool
Manager::signalWrite(address_space_impl aspace, hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    bool ret = Parent::signalWrite(aspace, addr);
    // POSTCONDITIONS

    return ret;
}

#if 0
gmacError_t
Manager::toIOBuffer(address_space_impl aspace, io_buffer_impl &buffer, size_t bufferOff, const hostptr_t addr, size_t size)
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
Manager::fromIOBuffer(address_space_impl aspace, hostptr_t addr, io_buffer_impl &buffer, size_t bufferOff, size_t size)
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
Manager::memcpy(address_space_impl aspace, hostptr_t dst, const hostptr_t src, size_t n)
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
Manager::memset(address_space_impl aspace, hostptr_t dst, int c, size_t n)
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

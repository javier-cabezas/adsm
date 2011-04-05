#ifdef USE_DBC

#include "core/IOBuffer.h"

#include "Manager.h"

namespace __dbc { namespace memory {

Manager::Manager() :
    __impl::memory::Manager()
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
    gmacError_t ret = __impl::memory::Manager::map(addr, size, prot);
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
    gmacError_t ret = __impl::memory::Manager::unmap(addr, size);
    // POSTCONDITIONS

    return ret;
}
#endif
gmacError_t
Manager::alloc(__impl::core::Mode &mode, hostptr_t *addr, size_t size)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Manager::alloc(mode, addr, size);
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
    gmacError_t ret = __impl::memory::Manager::globalAlloc(addr, size, hint);
    // POSTCONDITIONS

    return ret;
}
#endif

gmacError_t
Manager::free(__impl::core::Mode &mode, hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Manager::free(mode, addr);
    // POSTCONDITIONS

    return ret;
}

bool
Manager::read(__impl::core::Mode &mode, hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    bool ret = __impl::memory::Manager::read(mode, addr);
    // POSTCONDITIONS

    return ret;
}

bool
Manager::write(__impl::core::Mode &mode, hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    bool ret = __impl::memory::Manager::write(mode, addr);
    // POSTCONDITIONS

    return ret;
}


gmacError_t
Manager::toIOBuffer(__impl::core::Mode &mode, __impl::core::IOBuffer &buffer, size_t bufferOff, const hostptr_t addr, size_t size)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    REQUIRES(size > 0);
    REQUIRES(size <= buffer.size() - bufferOff);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Manager::toIOBuffer(mode, buffer, bufferOff, addr, size);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Manager::fromIOBuffer(__impl::core::Mode &mode, hostptr_t addr, __impl::core::IOBuffer &buffer, size_t bufferOff, size_t size)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    REQUIRES(size > 0);
    REQUIRES(size <= buffer.size() - bufferOff);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Manager::fromIOBuffer(mode, addr, buffer, bufferOff, size);
    // POSTCONDITIONS

    return ret;
}
#if 0
gmacError_t
Manager::memcpy(void * dst, const void * src, size_t n)
{
    // PRECONDITIONS
    REQUIRES(src != NULL);
    REQUIRES(dst != NULL);
    REQUIRES(n > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Manager::memcpy(dst, src, n);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Manager::memset(void * dst, int c, size_t n)
{
    // PRECONDITIONS
    REQUIRES(dst != NULL);
    REQUIRES(n > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Manager::memset(dst, c, n);
    // POSTCONDITIONS

    return ret;
}
#endif
}}

#endif

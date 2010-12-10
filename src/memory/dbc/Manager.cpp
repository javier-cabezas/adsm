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
Manager::alloc(hostptr_t *addr, size_t size)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Manager::alloc(addr, size);
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
Manager::free(hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Manager::free(addr);
    // POSTCONDITIONS

    return ret;
}

bool
Manager::read(hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    bool ret = __impl::memory::Manager::read(addr);
    // POSTCONDITIONS

    return ret;
}

bool
Manager::write(hostptr_t addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    bool ret = __impl::memory::Manager::write(addr);
    // POSTCONDITIONS

    return ret;
}


gmacError_t
Manager::toIOBuffer(__impl::core::IOBuffer &buffer, const hostptr_t addr, size_t size)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    REQUIRES(size > 0);
    REQUIRES(size <= buffer.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Manager::toIOBuffer(buffer, addr, size);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Manager::fromIOBuffer(hostptr_t addr, __impl::core::IOBuffer &buffer, size_t size)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    REQUIRES(size > 0);
    REQUIRES(size <= buffer.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::Manager::fromIOBuffer(addr, buffer, size);
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

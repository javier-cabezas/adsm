#include "Manager.h"

namespace gmac { namespace memory { namespace __dbc {

Manager::Manager() :
    __impl::Manager()
{
}

Manager::~Manager()
{
}

gmacError_t
Manager::map(void *addr, size_t size, GmacProtection prot)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::Manager::map(addr, size, prot);
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
    gmacError_t ret = __impl::Manager::unmap(addr, size);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Manager::alloc(void **addr, size_t size)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::Manager::alloc(addr, size);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Manager::globalAlloc(void **addr, size_t size, GmacGlobalMallocType hint)
{
    // PRECONDITIONS
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::Manager::globalAlloc(addr, size, hint);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Manager::free(void *addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::Manager::free(addr);
    // POSTCONDITIONS

    return ret;
}

bool
Manager::read(void *addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    bool ret = __impl::Manager::read(addr);
    // POSTCONDITIONS

    return ret;
}

bool
Manager::write(void *addr)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    // CALL IMPLEMENTATION
    bool ret = __impl::Manager::write(addr);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Manager::toIOBuffer(IOBuffer &buffer, const void *addr, size_t size)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::Manager::toIOBuffer(buffer, addr, size);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Manager::fromIOBuffer(void *addr, IOBuffer &buffer, size_t size)
{
    // PRECONDITIONS
    REQUIRES(addr != NULL);
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::Manager::toIOBuffer(buffer, addr, size);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Manager::memcpy(void * dst, const void * src, size_t n)
{
    // PRECONDITIONS
    REQUIRES(src != NULL);
    REQUIRES(dst != NULL);
    REQUIRES(n > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::Manager::memcpy(dst, src, n);
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
    gmacError_t ret = __impl::Manager::memset(dst, c, n);
    // POSTCONDITIONS

    return ret;
}

}}}

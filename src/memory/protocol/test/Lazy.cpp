#include "core/IOBuffer.h"

#include "Lazy.h"

namespace gmac { namespace memory { namespace protocol {

LazyTest::LazyTest(unsigned limit) :
    LazyImpl(limit)
{
}

LazyTest::~LazyTest()
{
}

gmacError_t
LazyTest::signalRead(const Object &obj, void *addr)
{
    // PRECONDITIONS
    REQUIRES(addr >= obj.addr());
    REQUIRES(addr < obj.end());

    // CALL IMPLEMENTATION
    gmacError_t ret = LazyImpl::signalRead(obj, addr);
    // POSTCONDITIONS
    return ret;
}

gmacError_t
LazyTest::signalWrite(const Object &obj, void *addr)
{
    // PRECONDITIONS
    REQUIRES(addr >= obj.addr());
    REQUIRES(addr < obj.end());

    // CALL IMPLEMENTATION
    gmacError_t ret = LazyImpl::signalWrite(obj, addr);
    // POSTCONDITIONS
    return ret;
}

gmacError_t
LazyTest::toIOBuffer(IOBuffer &buffer, unsigned bufferOff, const Object &obj, unsigned objectOff, size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(bufferOff + count <= buffer.size());
    REQUIRES(objectOff + count <= obj.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = LazyImpl::toIOBuffer(buffer, bufferOff, obj, objectOff, count);
    // POSTCONDITIONS
    return ret;
}

gmacError_t
LazyTest::fromIOBuffer(const Object &obj, unsigned objectOff, IOBuffer &buffer, unsigned bufferOff, size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(bufferOff + count <= buffer.size());
    REQUIRES(objectOff + count <= obj.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = LazyImpl::fromIOBuffer(obj, objectOff, buffer, bufferOff, count);
    // POSTCONDITIONS
    return ret;
}

gmacError_t
LazyTest::toPointer(void *dst, const Object &objSrc, unsigned objectOff, size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(objectOff + count <= objSrc.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = LazyImpl::toPointer(dst, objSrc, objectOff, count);
    // POSTCONDITIONS
    return ret;
}

gmacError_t
LazyTest::fromPointer(const Object &objDst, unsigned objectOff, const void *src, size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(objectOff + count <= objDst.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = LazyImpl::fromPointer(objDst, objectOff, src, count);
    // POSTCONDITIONS
    return ret;
}

}}}

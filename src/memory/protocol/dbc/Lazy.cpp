#ifdef USE_DBC

#include "core/IOBuffer.h"

#include "Lazy.h"

namespace __dbc { namespace memory { namespace protocol {

gmacError_t
Lazy::copyHostToDirty(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                      const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    // PRECONDITIONS
    REQUIRES(blockOffDst + count <= blockDst.size());
    REQUIRES(blockOffSrc + count <= blockSrc.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::protocol::Lazy::copyHostToDirty(objectDst, blockDst, blockOffDst,
                                                    objectSrc, blockSrc, blockOffSrc, count);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Lazy::copyHostToReadOnly(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                         const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    // PRECONDITIONS
    REQUIRES(blockOffDst + count <= blockDst.size());
    REQUIRES(blockOffSrc + count <= blockSrc.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::protocol::Lazy::copyHostToReadOnly(objectDst, blockDst, blockOffDst,
                                                       objectSrc, blockSrc, blockOffSrc, count);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Lazy::copyHostToInvalid(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                        const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    // PRECONDITIONS
    REQUIRES(blockOffDst + count <= blockDst.size());
    REQUIRES(blockOffSrc + count <= blockSrc.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::protocol::Lazy::copyHostToInvalid(objectDst, blockDst, blockOffDst,
                                                      objectSrc, blockSrc, blockOffSrc, count);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Lazy::copyAcceleratorToDirty(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                             const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    // PRECONDITIONS
    REQUIRES(blockOffDst + count <= blockDst.size());
    REQUIRES(blockOffSrc + count <= blockSrc.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::protocol::Lazy::copyAcceleratorToDirty(objectDst, blockDst, blockOffDst,
                                                           objectSrc, blockSrc, blockOffSrc, count);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Lazy::copyAcceleratorToReadOnly(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                                const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    // PRECONDITIONS
    REQUIRES(blockOffDst + count <= blockDst.size());
    REQUIRES(blockOffSrc + count <= blockSrc.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::protocol::Lazy::copyAcceleratorToReadOnly(objectDst, blockDst, blockOffDst,
                                                              objectSrc, blockSrc, blockOffSrc, count);
    // POSTCONDITIONS

    return ret;
}

gmacError_t
Lazy::copyAcceleratorToInvalid(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                               const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    // PRECONDITIONS
    REQUIRES(blockOffDst + count <= blockDst.size());
    REQUIRES(blockOffSrc + count <= blockSrc.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::protocol::Lazy::copyAcceleratorToInvalid(objectDst, blockDst, blockOffDst,
                                                             objectSrc, blockSrc, blockOffSrc, count);
    // POSTCONDITIONS

    return ret;
}

Lazy::Lazy(unsigned limit) :
    __impl::memory::protocol::Lazy(limit)
{
}

Lazy::~Lazy()
{
}

gmacError_t
Lazy::signalRead(const Object &obj, void *addr)
{
    // PRECONDITIONS
    REQUIRES(addr >= obj.addr());
    REQUIRES(addr < obj.end());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::protocol::Lazy::signalRead(obj, addr);
    // POSTCONDITIONS
    return ret;
}

gmacError_t
Lazy::signalWrite(const Object &obj, void *addr)
{
    // PRECONDITIONS
    REQUIRES(addr >= obj.addr());
    REQUIRES(addr < obj.end());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::protocol::Lazy::signalWrite(obj, addr);
    // POSTCONDITIONS
    return ret;
}

gmacError_t
Lazy::toIOBuffer(IOBuffer &buffer, unsigned bufferOff, const Object &obj, unsigned objectOff, size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(bufferOff + count <= buffer.size());
    REQUIRES(objectOff + count <= obj.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::protocol::Lazy::toIOBuffer(buffer, bufferOff, obj, objectOff, count);
    // POSTCONDITIONS
    return ret;
}

gmacError_t
Lazy::fromIOBuffer(const Object &obj, unsigned objectOff, IOBuffer &buffer, unsigned bufferOff, size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(bufferOff + count <= buffer.size());
    REQUIRES(objectOff + count <= obj.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::protocol::Lazy::fromIOBuffer(obj, objectOff, buffer, bufferOff, count);
    // POSTCONDITIONS
    return ret;
}

gmacError_t
Lazy::toPointer(void *dst, const Object &objSrc, unsigned objectOff, size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(objectOff + count <= objSrc.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::protocol::Lazy::toPointer(dst, objSrc, objectOff, count);
    // POSTCONDITIONS
    return ret;
}

gmacError_t
Lazy::fromPointer(const Object &objDst, unsigned objectOff, const void *src, size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(objectOff + count <= objDst.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::protocol::Lazy::fromPointer(objDst, objectOff, src, count);
    // POSTCONDITIONS
    return ret;
}

gmacError_t
Lazy::copy(const Object &objDst, unsigned offDst, const Object &objSrc, unsigned offSrc, size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(offDst + count <= objDst.size());
    REQUIRES(offSrc + count <= objSrc.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::protocol::Lazy::copy(objDst, offDst, objSrc, offSrc, count);
    // POSTCONDITIONS
    return ret;
}

gmacError_t
Lazy::memset(const Object &obj, unsigned objectOff, int c, size_t count)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(objectOff + count <= obj.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::protocol::Lazy::memset(obj, objectOff, c, count);
    // POSTCONDITIONS
    return ret;
}

}}}

#endif

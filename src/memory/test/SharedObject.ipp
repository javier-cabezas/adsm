#include "SharedObject.h"

namespace gmac { namespace memory{

template <typename T>
SharedObjectTest<T>::SharedObjectTest(size_t size, T init)
    : SharedObjectImpl<T>(size, init)
{
    REQUIRES(size > 0);
}

template <typename T>
SharedObjectTest<T>::~SharedObjectTest()
{
}

template<typename T>
void
SharedObjectTest<T>::init()
{
    // PRECONDITIONS

    // CALL IMPLEMENTATION
    SharedObjectImpl<T>::init();
    // POSTCONDITIONS
    ENSURES(SharedObjectTest<T>::addr() != NULL);
}

template<typename T>
void
SharedObjectTest<T>::fini()
{
    // PRECONDITIONS
    REQUIRES(SharedObjectTest<T>::addr() != NULL);
    // CALL IMPLEMENTATION
    SharedObjectImpl<T>::fini();
    // POSTCONDITIONS
}

template <typename T>
gmacError_t
SharedObjectTest<T>::toHost(Block &block) const
{
    // PRECONDITIONS

    // CALL IMPLEMENTATION
    gmacError_t ret = SharedObjectImpl<T>::toHost(block);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
SharedObjectTest<T>::toHost(Block &block, unsigned blockOff, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = SharedObjectImpl<T>::toHost(block, blockOff, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
SharedObjectTest<T>::toHostPointer(Block &block, unsigned blockOff, void *ptr, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = SharedObjectImpl<T>::toHostPointer(block, blockOff, ptr, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
SharedObjectTest<T>::toHostBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());
    REQUIRES(bufferOff + count <= buffer.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = SharedObjectImpl<T>::toHostBuffer(block, blockOff, buffer, bufferOff, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
SharedObjectTest<T>::toAccelerator(Block &block) const
{
    // PRECONDITIONS
    // CALL IMPLEMENTATION
    gmacError_t ret = SharedObjectImpl<T>::toAccelerator(block);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
SharedObjectTest<T>::toAccelerator(Block &block, unsigned blockOff, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = SharedObjectImpl<T>::toAccelerator(block, blockOff, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
SharedObjectTest<T>::toAcceleratorFromPointer(Block &block, unsigned blockOff, const void *ptr, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = SharedObjectImpl<T>::toAcceleratorFromPointer(block, blockOff, ptr, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
SharedObjectTest<T>::toAcceleratorFromBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());
    REQUIRES(bufferOff + count <= buffer.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = SharedObjectImpl<T>::toAcceleratorFromBuffer(block, blockOff, buffer, bufferOff, count);
    // POSTCONDITIONS
    return ret;
}

template<typename T>
Mode &
SharedObjectTest<T>::owner() const
{
    // PRECONDITIONS
    REQUIRES(SharedObjectImpl<T>::owner_ != NULL);

    // CALL IMPLEMENTATION
    Mode &ret = SharedObjectImpl<T>::owner();
    // POSTCONDITIONS
    return ret;
}

template<typename T>
gmacError_t
SharedObjectTest<T>::realloc(Mode &mode)
{
    // PRECONDITIONS
    REQUIRES(mode.id() != owner().id());

    // CALL IMPLEMENTATION
    gmacError_t ret = SharedObjectImpl<T>::realloc(mode);
    // POSTCONDITIONS
    ENSURES(mode.id() == owner().id());
    return ret;
}

}}

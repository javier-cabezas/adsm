#ifndef GMAC_MEMORY_TEST_SHAREDOBJECT_IPP_
#define GMAC_MEMORY_TEST_SHAREDOBJECT_IPP_

#include "SharedObject.h"

namespace gmac { namespace memory { namespace __dbc {

template <typename T>
SharedObject<T>::SharedObject(size_t size, void *cpuPtr, T init)
    : __impl::SharedObject<T>(size, cpuPtr, init)
{
    REQUIRES(size > 0);
}

template <typename T>
SharedObject<T>::~SharedObject()
{
}

template<typename T>
gmacError_t
SharedObject<T>::init()
{
    // PRECONDITIONS
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::SharedObject<T>::init();
    // POSTCONDITIONS
    ENSURES(ret == gmacErrorMemoryAllocation ||
            StateObject<T>::addr() != NULL);

    return ret;
}

template<typename T>
void
SharedObject<T>::fini()
{
    // PRECONDITIONS
    REQUIRES(StateObject<T>::addr() != NULL);
    // CALL IMPLEMENTATION
    __impl::SharedObject<T>::fini();
    // POSTCONDITIONS
}

template<typename T>
inline void *
SharedObject<T>::getAcceleratorAddr(void *addr) const
{
    // PRECONDITIONS
    REQUIRES(addr >= StateObject<T>::addr());
    REQUIRES(addr <  StateObject<T>::end());
    // CALL IMPLEMENTATION
    void *ret = __impl::SharedObject<T>::getAcceleratorAddr(addr);
    // POSTCONDITIONS
    return ret;
}



template <typename T>
gmacError_t
SharedObject<T>::toHost(Block &block) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= StateObject<T>::addr());
    REQUIRES(block.end() <= StateObject<T>::end());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::SharedObject<T>::toHost(block);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
SharedObject<T>::toHost(Block &block, unsigned blockOff, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= StateObject<T>::addr());
    REQUIRES(block.end() <= StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::SharedObject<T>::toHost(block, blockOff, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
SharedObject<T>::toHostPointer(Block &block, unsigned blockOff, void *ptr, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= StateObject<T>::addr());
    REQUIRES(block.end() <= StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::SharedObject<T>::toHostPointer(block, blockOff, ptr, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
SharedObject<T>::toHostBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= StateObject<T>::addr());
    REQUIRES(block.end() <= StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());
    REQUIRES(bufferOff + count <= buffer.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::SharedObject<T>::toHostBuffer(block, blockOff, buffer, bufferOff, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
SharedObject<T>::toAccelerator(Block &block) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= StateObject<T>::addr());
    REQUIRES(block.end() <= StateObject<T>::end());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::SharedObject<T>::toAccelerator(block);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
SharedObject<T>::toAccelerator(Block &block, unsigned blockOff, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= StateObject<T>::addr());
    REQUIRES(block.end() <= StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::SharedObject<T>::toAccelerator(block, blockOff, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
SharedObject<T>::toAcceleratorFromPointer(Block &block, unsigned blockOff, const void *ptr, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= StateObject<T>::addr());
    REQUIRES(block.end() <= StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::SharedObject<T>::toAcceleratorFromPointer(block, blockOff, ptr, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
SharedObject<T>::toAcceleratorFromBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= StateObject<T>::addr());
    REQUIRES(block.end() <= StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());
    REQUIRES(bufferOff + count <= buffer.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::SharedObject<T>::toAcceleratorFromBuffer(block, blockOff, buffer, bufferOff, count);
    // POSTCONDITIONS
    return ret;
}

template<typename T>
Mode &
SharedObject<T>::owner() const
{
    // PRECONDITIONS
    REQUIRES(__impl::SharedObject<T>::owner_ != NULL);

    // CALL IMPLEMENTATION
    Mode &ret = __impl::SharedObject<T>::owner();
    // POSTCONDITIONS
    return ret;
}

template<typename T>
gmacError_t
SharedObject<T>::free()
{
    // PRECONDITIONS
    REQUIRES(__impl::SharedObject<T>::accBlock_ != NULL);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::SharedObject<T>::free();
    // POSTCONDITIONS
    ENSURES(__impl::SharedObject<T>::accBlock_ == NULL);
    return ret;
}

template<typename T>
gmacError_t
SharedObject<T>::realloc(Mode &mode)
{
    // PRECONDITIONS
    REQUIRES(mode.id() != owner().id());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::SharedObject<T>::realloc(mode);
    // POSTCONDITIONS
    ENSURES(mode.id() == owner().id());
    return ret;
}

}}}

#endif

#ifndef GMAC_MEMORY_TEST_SHAREDOBJECT_IPP_
#define GMAC_MEMORY_TEST_SHAREDOBJECT_IPP_

#include "SharedObject.h"

namespace __dbc { namespace memory {

template <typename T>
SharedObject<T>::SharedObject(size_t size, void *cpuPtr, T init)
    : __impl::memory::SharedObject<T>(size, cpuPtr, init)
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
    gmacError_t ret = __impl::memory::SharedObject<T>::init();
    // POSTCONDITIONS
    ENSURES(ret == gmacErrorMemoryAllocation ||
            __impl::memory::StateObject<T>::addr() != NULL);

    return ret;
}

template<typename T>
void
SharedObject<T>::fini()
{
    // PRECONDITIONS
    REQUIRES(__impl::memory::StateObject<T>::addr() != NULL);
    // CALL IMPLEMENTATION
    __impl::memory::SharedObject<T>::fini();
    // POSTCONDITIONS
}

template<typename T>
inline void *
SharedObject<T>::getAcceleratorAddr(void *addr) const
{
    // PRECONDITIONS
    REQUIRES(addr >= __impl::memory::StateObject<T>::addr());
    REQUIRES(addr <  __impl::memory::StateObject<T>::end());
    // CALL IMPLEMENTATION
    void *ret = __impl::memory::SharedObject<T>::getAcceleratorAddr(addr);
    // POSTCONDITIONS
    return ret;
}



template <typename T>
gmacError_t
SharedObject<T>::toHost(__impl::memory::Block &block) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= __impl::memory::StateObject<T>::addr());
    REQUIRES(block.end() <= __impl::memory::StateObject<T>::end());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::SharedObject<T>::toHost(block);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);
    return ret;
}

template <typename T>
gmacError_t
SharedObject<T>::toHost(__impl::memory::Block &block, unsigned blockOff, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= __impl::memory::StateObject<T>::addr());
    REQUIRES(block.end() <= __impl::memory::StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::SharedObject<T>::toHost(block, blockOff, count);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);
    return ret;
}

template <typename T>
gmacError_t
SharedObject<T>::toHostPointer(__impl::memory::Block &block, unsigned blockOff, void *ptr, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= __impl::memory::StateObject<T>::addr());
    REQUIRES(block.end() <= __impl::memory::StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::SharedObject<T>::toHostPointer(block, blockOff, ptr, count);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);
    return ret;
}

template <typename T>
gmacError_t
SharedObject<T>::toHostBuffer(__impl::memory::Block &block, unsigned blockOff, __impl::core::IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= __impl::memory::StateObject<T>::addr());
    REQUIRES(block.end() <= __impl::memory::StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());
    REQUIRES(bufferOff + count <= buffer.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::SharedObject<T>::toHostBuffer(block, blockOff, buffer, bufferOff, count);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);
    return ret;
}

template <typename T>
gmacError_t
SharedObject<T>::toAccelerator(__impl::memory::Block &block) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= __impl::memory::StateObject<T>::addr());
    REQUIRES(block.end() <= __impl::memory::StateObject<T>::end());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::SharedObject<T>::toAccelerator(block);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);
    return ret;
}

template <typename T>
gmacError_t
SharedObject<T>::toAccelerator(__impl::memory::Block &block, unsigned blockOff, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= __impl::memory::StateObject<T>::addr());
    REQUIRES(block.end() <= __impl::memory::StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::SharedObject<T>::toAccelerator(block, blockOff, count);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);
    return ret;
}

template <typename T>
gmacError_t
SharedObject<T>::toAcceleratorFromPointer(__impl::memory::Block &block, unsigned blockOff, const void *ptr, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= __impl::memory::StateObject<T>::addr());
    REQUIRES(block.end() <= __impl::memory::StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::SharedObject<T>::toAcceleratorFromPointer(block, blockOff, ptr, count);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);
    return ret;
}

template <typename T>
gmacError_t
SharedObject<T>::toAcceleratorFromBuffer(__impl::memory::Block &block, unsigned blockOff, __impl::core::IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= __impl::memory::StateObject<T>::addr());
    REQUIRES(block.end() <= __impl::memory::StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());
    REQUIRES(bufferOff + count <= buffer.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::SharedObject<T>::toAcceleratorFromBuffer(block, blockOff, buffer, bufferOff, count);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);
    return ret;
}

template<typename T>
__impl::core::Mode &
SharedObject<T>::owner() const
{
    // PRECONDITIONS
    REQUIRES(__impl::memory::SharedObject<T>::owner_ != NULL);

    // CALL IMPLEMENTATION
    __impl::core::Mode &ret = __impl::memory::SharedObject<T>::owner();
    // POSTCONDITIONS
    return ret;
}

template<typename T>
gmacError_t
SharedObject<T>::free()
{
    // PRECONDITIONS
    REQUIRES(__impl::memory::SharedObject<T>::accBlock_ != NULL);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::SharedObject<T>::free();
    // POSTCONDITIONS
    ENSURES(__impl::memory::SharedObject<T>::accBlock_ == NULL);
    return ret;
}

template<typename T>
gmacError_t
SharedObject<T>::realloc(__impl::core::Mode &mode)
{
    // PRECONDITIONS
    REQUIRES(mode.id() != owner().id());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::SharedObject<T>::realloc(mode);
    // POSTCONDITIONS
    ENSURES(mode.id() == owner().id());
    return ret;
}

}}

#endif

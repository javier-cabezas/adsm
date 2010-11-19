#ifndef GMAC_MEMORY_TEST_REPLICATEDOBJECT_IPP_
#define GMAC_MEMORY_TEST_REPLICATEDOBJECT_IPP_

#include "trace/Tracer.h"

namespace __dbc { namespace memory {

template <typename T>
ReplicatedObject<T>::ReplicatedObject(size_t size, T init)
    : __impl::memory::ReplicatedObject<T>(size, init)
{
    REQUIRES(size > 0);
}

template <typename T>
ReplicatedObject<T>::~ReplicatedObject()
{
}

template<typename T>
gmacError_t
ReplicatedObject<T>::init()
{
    // PRECONDITIONS
    ENSURES(__impl::memory::StateObject<T>::addr() == NULL);

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::ReplicatedObject<T>::init();
    // POSTCONDITIONS
    ENSURES(gmacErrorMemoryAllocation ||
            __impl::memory::StateObject<T>::addr() != NULL);

    return ret;
}

template<typename T>
void
ReplicatedObject<T>::fini()
{
    // PRECONDITIONS
    REQUIRES(__impl::memory::StateObject<T>::addr() != NULL);
    // CALL IMPLEMENTATION
    __impl::memory::ReplicatedObject<T>::fini();
    // POSTCONDITIONS
}

template<typename T>
inline void *
ReplicatedObject<T>::getAcceleratorAddr(void *addr) const
{
    // PRECONDITIONS
    REQUIRES(addr >= __impl::memory::StateObject<T>::addr());
    REQUIRES(addr <  __impl::memory::StateObject<T>::end());
    // CALL IMPLEMENTATION
    void *ret = __impl::memory::ReplicatedObject<T>::getAcceleratorAddr(addr);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toHost(__impl::memory::Block &block) const
{
    // Function not valid for Replicated Objects
    // PRECONDITIONS
    REQUIRES(0);
    gmacError_t ret = __impl::memory::ReplicatedObject<T>::toHost(block);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toHost(__impl::memory::Block &block, unsigned blockOff, size_t count) const
{
    // Function not valid for Replicated Objects
    // PRECONDITIONS
    REQUIRES(0);
    gmacError_t ret = __impl::memory::ReplicatedObject<T>::toHost(block, blockOff, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toHostPointer(__impl::memory::Block &block, unsigned blockOff, void *ptr, size_t count) const
{
    // Function not valid for Replicated Objects
    // PRECONDITIONS
    REQUIRES(0);
    gmacError_t ret = __impl::memory::ReplicatedObject<T>::toHostPointer(block, blockOff, ptr, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toHostBuffer(__impl::memory::Block &block, unsigned blockOff, __impl::core::IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    // Function not valid for Replicated Objects
    // PRECONDITIONS
    REQUIRES(0);
    gmacError_t ret = __impl::memory::ReplicatedObject<T>::toHostBuffer(block, blockOff, buffer, bufferOff, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toAccelerator(__impl::memory::Block &block) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= __impl::memory::StateObject<T>::addr());
    REQUIRES(block.end() <= __impl::memory::StateObject<T>::end());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::ReplicatedObject<T>::toAccelerator(block);
    // POSTCONDITIONS

    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toAccelerator(__impl::memory::Block &block, unsigned blockOff, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= __impl::memory::StateObject<T>::addr());
    REQUIRES(block.end() <= __impl::memory::StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::ReplicatedObject<T>::toAccelerator(block, blockOff, count);
    // POSTCONDITIONS

    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toAcceleratorFromPointer(__impl::memory::Block &block, unsigned blockOff, const void *ptr, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= __impl::memory::StateObject<T>::addr());
    REQUIRES(block.end() <= __impl::memory::StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::ReplicatedObject<T>::toAcceleratorFromPointer(block, blockOff, ptr, count);
    // POSTCONDITIONS

    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toAcceleratorFromBuffer(__impl::memory::Block &block, unsigned blockOff, __impl::core::IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    // PRECONDITIONS
    REQUIRES(block.addr() >= __impl::memory::StateObject<T>::addr());
    REQUIRES(block.end() <= __impl::memory::StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());
    REQUIRES(bufferOff + count <= buffer.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::ReplicatedObject<T>::toAcceleratorFromBuffer(block, blockOff, buffer, bufferOff, count);
    // POSTCONDITIONS

    return ret;
}

template<typename T>
__impl::core::Mode &
ReplicatedObject<T>::owner() const
{
    // PRECONDITIONS

    // CALL IMPLEMENTATION
    __impl::core::Mode &ret = __impl::memory::ReplicatedObject<T>::owner();
    // POSTCONDITIONS
    ENSURES(&ret != NULL);
    return ret;
}

template<typename T>
gmacError_t ReplicatedObject<T>::addOwner(__impl::core::Mode &mode)
{
    // PRECONDITIONS
    REQUIRES(__impl::memory::ReplicatedObject<T>::accelerators_.find(&mode) == __impl::memory::ReplicatedObject<T>::accelerators_.end());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::ReplicatedObject<T>::addOwner(mode);
    // POSTCONDITIONS
    ENSURES(__impl::memory::ReplicatedObject<T>::accelerators_.find(&mode) != __impl::memory::ReplicatedObject<T>::accelerators_.end());
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::removeOwner(__impl::core::Mode &mode)
{
    // PRECONDITIONS
    REQUIRES(__impl::memory::ReplicatedObject<T>::accelerators_.find(&mode) != __impl::memory::ReplicatedObject<T>::accelerators_.end());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::memory::ReplicatedObject<T>::removeOwner(mode);
    // POSTCONDITIONS
    ENSURES(__impl::memory::ReplicatedObject<T>::accelerators_.find(&mode) == __impl::memory::ReplicatedObject<T>::accelerators_.end());
    return ret;

}

}}

#endif

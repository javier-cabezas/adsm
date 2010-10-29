#ifndef GMAC_MEMORY_TEST_REPLICATEDOBJECT_IPP_
#define GMAC_MEMORY_TEST_REPLICATEDOBJECT_IPP_

#include "ReplicatedObject.h"

namespace gmac { namespace memory { namespace __dbc {

template <typename T>
ReplicatedObject<T>::ReplicatedObject(size_t size, T init)
    : __impl::ReplicatedObject<T>(size, init)
{
    REQUIRES(size > 0);
}

template <typename T>
ReplicatedObject<T>::~ReplicatedObject()
{
}

template<typename T>
void
ReplicatedObject<T>::init()
{
    // PRECONDITIONS
    ENSURES(StateObject<T>::addr() == NULL);

    // CALL IMPLEMENTATION
    __impl::ReplicatedObject<T>::init();
    // POSTCONDITIONS
    ENSURES(StateObject<T>::addr() != NULL);
}

template<typename T>
void
ReplicatedObject<T>::fini()
{
    // PRECONDITIONS
    REQUIRES(StateObject<T>::addr() != NULL);
    // CALL IMPLEMENTATION
    __impl::ReplicatedObject<T>::fini();
    // POSTCONDITIONS
}

template<typename T>
inline void *
ReplicatedObject<T>::getAcceleratorAddr(void *addr) const
{
    // PRECONDITIONS
    REQUIRES(addr >= StateObject<T>::addr());
    REQUIRES(addr <  StateObject<T>::end());
    // CALL IMPLEMENTATION
    void *ret = __impl::ReplicatedObject<T>::getAcceleratorAddr(addr);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toHost(Block &block) const
{
    // Function not valid for Replicated Objects
    // PRECONDITIONS
    REQUIRES(0);
    gmacError_t ret = __impl::ReplicatedObject<T>::toHost(block);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toHost(Block &block, unsigned blockOff, size_t count) const
{
    // Function not valid for Replicated Objects
    // PRECONDITIONS
    REQUIRES(0);
    gmacError_t ret = __impl::ReplicatedObject<T>::toHost(block, blockOff, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toHostPointer(Block &block, unsigned blockOff, void *ptr, size_t count) const
{
    // Function not valid for Replicated Objects
    // PRECONDITIONS
    REQUIRES(0);
    gmacError_t ret = __impl::ReplicatedObject<T>::toHostPointer(block, blockOff, ptr, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toHostBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    // Function not valid for Replicated Objects
    // PRECONDITIONS
    REQUIRES(0);
    gmacError_t ret = __impl::ReplicatedObject<T>::toHostBuffer(block, blockOff, buffer, bufferOff, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toAccelerator(Block &block) const
{
    trace::Function::start("ReplicatedObject", "toAccelerator");
    // PRECONDITIONS
    REQUIRES(block.addr() >= StateObject<T>::addr());
    REQUIRES(block.end() <= StateObject<T>::end());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::ReplicatedObject<T>::toAccelerator(block);
    // POSTCONDITIONS
    trace::Function::end("ReplicatedObject");
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toAccelerator(Block &block, unsigned blockOff, size_t count) const
{
    trace::Function::start("ReplicatedObject", "toAccelerator");
    // PRECONDITIONS
    REQUIRES(block.addr() >= StateObject<T>::addr());
    REQUIRES(block.end() <= StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::ReplicatedObject<T>::toAccelerator(block, blockOff, count);
    // POSTCONDITIONS
    trace::Function::end("ReplicatedObject");
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toAcceleratorFromPointer(Block &block, unsigned blockOff, const void *ptr, size_t count) const
{
    trace::Function::start("ReplicatedObject", "toAcceleratorFromPointer");
    // PRECONDITIONS
    REQUIRES(block.addr() >= StateObject<T>::addr());
    REQUIRES(block.end() <= StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::ReplicatedObject<T>::toAcceleratorFromPointer(block, blockOff, ptr, count);
    // POSTCONDITIONS
    trace::Function::end("ReplicatedObject");
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObject<T>::toAcceleratorFromBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    trace::Function::start("ReplicatedObject", "toAcceleratorFromBuffer");
    // PRECONDITIONS
    REQUIRES(block.addr() >= StateObject<T>::addr());
    REQUIRES(block.end() <= StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());
    REQUIRES(bufferOff + count <= buffer.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::ReplicatedObject<T>::toAcceleratorFromBuffer(block, blockOff, buffer, bufferOff, count);
    // POSTCONDITIONS
    trace::Function::end("ReplicatedObject");
    return ret;
}

template<typename T>
Mode &
ReplicatedObject<T>::owner() const
{
    // PRECONDITIONS

    // CALL IMPLEMENTATION
    Mode &ret = __impl::ReplicatedObject<T>::owner();
    // POSTCONDITIONS
    ENSURES(&ret != NULL);
    return ret;
}

template<typename T>
gmacError_t ReplicatedObject<T>::addOwner(Mode &mode)
{
    // PRECONDITIONS
    REQUIRES(__impl::ReplicatedObject<T>::accelerators_.find(&mode) == __impl::ReplicatedObject<T>::accelerators_.end());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::ReplicatedObject<T>::addOwner(mode);
    // POSTCONDITIONS
    ENSURES(__impl::ReplicatedObject<T>::accelerators_.find(&mode) != __impl::ReplicatedObject<T>::accelerators_.end());
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::removeOwner(Mode &mode)
{
    // PRECONDITIONS
    REQUIRES(__impl::ReplicatedObject<T>::accelerators_.find(&mode) != __impl::ReplicatedObject<T>::accelerators_.end());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::ReplicatedObject<T>::removeOwner(mode);
    // POSTCONDITIONS
    ENSURES(__impl::ReplicatedObject<T>::accelerators_.find(&mode) == __impl::ReplicatedObject<T>::accelerators_.end());
    return ret;

}

}}}

#endif

#ifndef GMAC_MEMORY_TEST_REPLICATEDOBJECT_IPP_
#define GMAC_MEMORY_TEST_REPLICATEDOBJECT_IPP_

#include "ReplicatedObject.h"

namespace gmac { namespace memory{

template <typename T>
ReplicatedObjectTest<T>::ReplicatedObjectTest(size_t size, T init)
    : ReplicatedObjectImpl<T>(size, init)
{
    REQUIRES(size > 0);
}

template <typename T>
ReplicatedObjectTest<T>::~ReplicatedObjectTest()
{
}

template<typename T>
void
ReplicatedObjectTest<T>::init()
{
    // PRECONDITIONS
    ENSURES(StateObject<T>::addr() == NULL);

    // CALL IMPLEMENTATION
    ReplicatedObjectImpl<T>::init();
    // POSTCONDITIONS
    ENSURES(StateObject<T>::addr() != NULL);
}

template<typename T>
void
ReplicatedObjectTest<T>::fini()
{
    // PRECONDITIONS
    REQUIRES(StateObject<T>::addr() != NULL);
    // CALL IMPLEMENTATION
    ReplicatedObjectImpl<T>::fini();
    // POSTCONDITIONS
}

template<typename T>
inline void *
ReplicatedObjectTest<T>::getAcceleratorAddr(void *addr) const
{
    // PRECONDITIONS
    REQUIRES(addr >= StateObject<T>::addr());
    REQUIRES(addr <  StateObject<T>::end());
    // CALL IMPLEMENTATION
    void *ret = ReplicatedObjectImpl<T>::getAcceleratorAddr(addr);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObjectTest<T>::toHost(Block &block) const
{
    // Function not valid for Replicated Objects
    // PRECONDITIONS
    REQUIRES(0);
    gmacError_t ret = ReplicatedObjectImpl<T>::toHost(block);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObjectTest<T>::toHost(Block &block, unsigned blockOff, size_t count) const
{
    // Function not valid for Replicated Objects
    // PRECONDITIONS
    REQUIRES(0);
    gmacError_t ret = ReplicatedObjectImpl<T>::toHost(block, blockOff, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObjectTest<T>::toHostPointer(Block &block, unsigned blockOff, void *ptr, size_t count) const
{
    // Function not valid for Replicated Objects
    // PRECONDITIONS
    REQUIRES(0);
    gmacError_t ret = ReplicatedObjectImpl<T>::toHostPointer(block, blockOff, ptr, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObjectTest<T>::toHostBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    // Function not valid for Replicated Objects
    // PRECONDITIONS
    REQUIRES(0);
    gmacError_t ret = ReplicatedObjectImpl<T>::toHostBuffer(block, blockOff, buffer, bufferOff, count);
    // POSTCONDITIONS
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObjectTest<T>::toAccelerator(Block &block) const
{
    trace::Function::start("ReplicatedObject", "toAccelerator");
    // PRECONDITIONS
    REQUIRES(block.addr() >= StateObject<T>::addr());
    REQUIRES(block.end() <= StateObject<T>::end());
    // CALL IMPLEMENTATION
    gmacError_t ret = ReplicatedObjectImpl<T>::toAccelerator(block);
    // POSTCONDITIONS
    trace::Function::end("ReplicatedObject");
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObjectTest<T>::toAccelerator(Block &block, unsigned blockOff, size_t count) const
{
    trace::Function::start("ReplicatedObject", "toAccelerator");
    // PRECONDITIONS
    REQUIRES(block.addr() >= StateObject<T>::addr());
    REQUIRES(block.end() <= StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = ReplicatedObjectImpl<T>::toAccelerator(block, blockOff, count);
    // POSTCONDITIONS
    trace::Function::end("ReplicatedObject");
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObjectTest<T>::toAcceleratorFromPointer(Block &block, unsigned blockOff, const void *ptr, size_t count) const
{
    trace::Function::start("ReplicatedObject", "toAcceleratorFromPointer");
    // PRECONDITIONS
    REQUIRES(block.addr() >= StateObject<T>::addr());
    REQUIRES(block.end() <= StateObject<T>::end());
    REQUIRES(count > 0);
    REQUIRES(blockOff < block.size());
    REQUIRES(blockOff + count <= block.size());

    // CALL IMPLEMENTATION
    gmacError_t ret = ReplicatedObjectImpl<T>::toAcceleratorFromPointer(block, blockOff, ptr, count);
    // POSTCONDITIONS
    trace::Function::end("ReplicatedObject");
    return ret;
}

template <typename T>
gmacError_t
ReplicatedObjectTest<T>::toAcceleratorFromBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
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
    gmacError_t ret = ReplicatedObjectImpl<T>::toAcceleratorFromBuffer(block, blockOff, buffer, bufferOff, count);
    // POSTCONDITIONS
    trace::Function::end("ReplicatedObject");
    return ret;
}

template<typename T>
Mode &
ReplicatedObjectTest<T>::owner() const
{
    // PRECONDITIONS

    // CALL IMPLEMENTATION
    Mode &ret = ReplicatedObjectImpl<T>::owner();
    // POSTCONDITIONS
    ENSURES(&ret != NULL);
    return ret;
}

template<typename T>
gmacError_t ReplicatedObjectTest<T>::addOwner(Mode &mode)
{
    // PRECONDITIONS
    REQUIRES(ReplicatedObjectImpl<T>::accelerators_.find(&mode) == ReplicatedObjectImpl<T>::accelerators_.end());
    // CALL IMPLEMENTATION
    gmacError_t ret = ReplicatedObjectImpl<T>::addOwner(mode);
    // POSTCONDITIONS
    ENSURES(ReplicatedObjectImpl<T>::accelerators_.find(&mode) != ReplicatedObjectImpl<T>::accelerators_.end());
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObjectTest<T>::removeOwner(Mode &mode)
{
    // PRECONDITIONS
    REQUIRES(ReplicatedObjectImpl<T>::accelerators_.find(&mode) != ReplicatedObjectImpl<T>::accelerators_.end());
    // CALL IMPLEMENTATION
    gmacError_t ret = ReplicatedObjectImpl<T>::removeOwner(mode);
    // POSTCONDITIONS
    ENSURES(ReplicatedObjectImpl<T>::accelerators_.find(&mode) == ReplicatedObjectImpl<T>::accelerators_.end());
    return ret;

}

}}

#endif

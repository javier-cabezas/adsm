#ifndef __MEMORY_SHAREDOBJECT_IPP
#define __MEMORY_SHAREDOBJECT_IPP

namespace gmac { namespace memory {

#ifdef USE_VM
#include "Bitmap.h"
#endif

template<typename T>
inline SharedObjectImpl<T>::SharedObjectImpl(size_t size, T init) :
    StateObject<T>(size, init),
    owner_(&Mode::current()),
    accBlock_(NULL)
{
    gmacError_t ret = gmacSuccess;
    void *accAddr = NULL;
    // Allocate accelerator and host memory
    ret = owner_->malloc(&accAddr, size, paramPageSize);
    if(ret != gmacSuccess) {
        StateObject<T>::addr_ = NULL;
        return;
    }
#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->dirtyBitmap();
    bitmap.newRange(accAddr, size);
#endif
    // Create memory blocks
    accBlock_ = new AcceleratorBlock(*owner_, accAddr, StateObject<T>::size_);

}


template<typename T>
inline SharedObjectImpl<T>::~SharedObjectImpl()
{
    if(StateObject<T>::addr_ == NULL) { return; }
    void *devAddr = accBlock_->addr();
    delete accBlock_;
    owner_->free(devAddr);
#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->dirtyBitmap();
    bitmap.removeRange(devAddr, StateObject<T>::size_);
#endif

    trace("Destroying Shared Object %p (%zd bytes)", StateObject<T>::addr_);
}

template<typename T>
inline void SharedObjectImpl<T>::init()
{

#ifdef USE_MMAP
    StateObject<T>::addr_ = StateObject<T>::map(device, StateObject<T>::size_);
#else
    StateObject<T>::addr_ = StateObject<T>::map(NULL, StateObject<T>::size_);
#endif

    if(StateObject<T>::addr_ == NULL) {
        return;
    }

    trace("Shared Object %p (%zd bytes) @ %p initialized", StateObject<T>::addr_, StateObject<T>::size_, accBlock_->addr());
    StateObject<T>::setupSystem();
}

template<typename T>
inline void SharedObjectImpl<T>::fini()
{
    StateObject<T>::unmap(StateObject<T>::addr_, StateObject<T>::size_);
}


template<typename T>
inline void *SharedObjectImpl<T>::getAcceleratorAddr(void *addr) const
{
    unsigned offset = (unsigned long)addr - (unsigned long)StateObject<T>::addr_;
    void *ret = accBlock_->addr() + offset;
    return ret;
}

template<typename T>
inline gmacError_t SharedObjectImpl<T>::toHost(Block &block) const
{
    unsigned offset = block.addr() - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().copyToHost(block.addr(), accBlock_->addr() + offset, block.size());
    return ret;
}

template<typename T>
inline gmacError_t SharedObjectImpl<T>::toHost(Block &block, unsigned blockOff, size_t count) const
{
    unsigned offset = block.addr() + blockOff - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().copyToHost(block.addr() + blockOff, accBlock_->addr() + offset, count);

    return ret;
}

template<typename T>
inline gmacError_t SharedObjectImpl<T>::toHostPointer(Block &block, unsigned blockOff, void * ptr, size_t count) const
{
    unsigned offset = block.addr() + blockOff - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().copyToHost(ptr, accBlock_->addr() + offset, count);

    return ret;
}

template<typename T>
inline gmacError_t SharedObjectImpl<T>::toHostBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    unsigned offset = block.addr() + blockOff - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().acceleratorToBuffer(buffer, accBlock_->addr() + offset, bufferOff, count);

    return ret;
}

template<typename T>
inline gmacError_t SharedObjectImpl<T>::toAccelerator(Block &block) const
{
    unsigned offset = block.addr() - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().copyToAccelerator(accBlock_->addr() + offset, block.addr(), block.size());
    return ret;
}

template<typename T>
inline gmacError_t SharedObjectImpl<T>::toAccelerator(Block &block, unsigned blockOff, size_t count) const
{
    unsigned offset = block.addr() + blockOff - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().copyToAccelerator(accBlock_->addr() + offset, block.addr() + blockOff, count);
    return ret;
}

template<typename T>
inline gmacError_t SharedObjectImpl<T>::toAcceleratorFromPointer(Block &block, unsigned blockOff, const void * ptr, size_t count) const
{
    unsigned offset = block.addr() + blockOff - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().copyToAccelerator(accBlock_->addr() + offset, ptr, count);

    return ret;
}

template<typename T>
inline gmacError_t SharedObjectImpl<T>::toAcceleratorFromBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    unsigned offset = block.addr() + blockOff - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().bufferToAccelerator(accBlock_->addr() + offset, buffer, bufferOff, count);

    return ret;
}


template<typename T>
gmacError_t SharedObjectImpl<T>::free()
{
#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->dirtyBitmap();
    bitmap.removeRange(accBlock_->addr(), StateObject<T>::size_);
#endif
    gmacError_t ret;
    ret = owner_->free(accBlock_->addr());
    delete accBlock_;
    accBlock_ = NULL;
    return ret;
}

template<typename T>
gmacError_t SharedObjectImpl<T>::realloc(Mode &mode)
{
    void *device = NULL;
    // Allocate device and host memory
    gmacError_t ret = mode.malloc(&device, StateObject<T>::size_, paramPageSize);
    if(ret != gmacSuccess) {
        StateObject<T>::addr_ = NULL;
        return gmacErrorInsufficientAcceleratorMemory;
    }

    trace("Reallocating object %p -> %p\n", accBlock_->addr(), device);

#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->dirtyBitmap();
    bitmap.newRange(device, StateObject<T>::size_);
#endif
    owner_ = &mode;
    accBlock_ = new AcceleratorBlock(mode, device, StateObject<T>::size_);
    return gmacSuccess;
}

template<typename T>
bool SharedObjectImpl<T>::isLocal() const
{
    return true;
}

template<typename T>
bool SharedObjectImpl<T>::isInAccelerator() const
{
    return true;
}

}}

#endif

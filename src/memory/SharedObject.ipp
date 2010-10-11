#ifndef __MEMORY_SHAREDOBJECT_IPP
#define __MEMORY_SHAREDOBJECT_IPP

namespace gmac { namespace memory {

#ifdef USE_VM
#include "Bitmap.h"
#endif

template<typename T>
inline SharedObject<T>::SharedObject(size_t size, T init) :
    StateObject<T>(size, init),
    owner_(&Mode::current()),
    accBlock_(NULL)
{
    gmacError_t ret = gmacSuccess;
    void *device = NULL;
    // Allocate device and host memory
    ret = owner_->malloc(&device, size, paramPageSize);
    if(ret != gmacSuccess) {
        StateObject<T>::addr_ = NULL;
        return;
    }
#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->dirtyBitmap();
    bitmap.newRange(device, size);
#endif
    // Create memory blocks
    accBlock_ = new AcceleratorBlock(*owner_, device, StateObject<T>::size_);

}


template<typename T>
inline SharedObject<T>::~SharedObject()
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
inline void SharedObject<T>::init()
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
inline void SharedObject<T>::fini()
{
    StateObject<T>::unmap(StateObject<T>::addr_, StateObject<T>::size_);
}


template<typename T>
inline void *SharedObject<T>::device(void *addr) const
{
    off_t offset = (unsigned long)addr - (unsigned long)StateObject<T>::addr_;
    void *ret = accBlock_->addr() + offset;
    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toHost(Block &block) const
{
    off_t off = block.addr() - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().copyToHost(block.addr(), accBlock_->addr() + off, block.size());
    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toHost(Block &block, unsigned blockOff, size_t count) const
{
    assertion(block.addr() + blockOff + count <= block.end());
    off_t off = block.addr() + blockOff - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().copyToHost(block.addr() + blockOff, accBlock_->addr() + off, count);

    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toHostPointer(Block &block, unsigned blockOff, void * ptr, size_t count) const
{
    assertion(block.addr() + blockOff + count <= block.end());
    off_t off = block.addr() + blockOff - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().copyToHost(ptr, accBlock_->addr() + off, count);

    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toHostBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    assertion(block.addr() + blockOff + count <= block.end());
    assertion(buffer.addr() + bufferOff + count <= buffer.end());
    off_t off = block.addr() + blockOff - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().deviceToBuffer(buffer, accBlock_->addr() + off, bufferOff, count);

    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toAccelerator(Block &block) const
{
    off_t off = block.addr() - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().copyToDevice(accBlock_->addr() + off, block.addr(), block.size());
    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toAccelerator(Block &block, unsigned blockOff, size_t count) const
{
    assertion(block.addr() + blockOff + count <= block.end());
    off_t off = block.addr() + blockOff - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().copyToDevice(accBlock_->addr() + off, block.addr() + blockOff, count);
    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toAcceleratorFromPointer(Block &block, unsigned blockOff, const void * ptr, size_t count) const
{
    assertion(block.addr() + blockOff + count <= block.end());
    off_t off = block.addr() + blockOff - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().copyToDevice(accBlock_->addr() + off, ptr, count);

    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toAcceleratorFromBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    assertion(block.addr() + blockOff + count <= block.end());
    assertion(buffer.addr() + bufferOff + count <= buffer.end());
    off_t off = block.addr() + blockOff - StateObject<T>::addr();
    gmacError_t ret = accBlock_->owner().bufferToDevice(accBlock_->addr() + off, buffer, bufferOff, count);

    return ret;
}


template<typename T>
gmacError_t SharedObject<T>::free()
{
#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->dirtyBitmap();
    bitmap.removeRange(accBlock_->addr(), StateObject<T>::size_);
#endif
    gmacError_t ret;
    ret = owner_->free(accBlock_->addr());
    delete accBlock_;
    return ret;
}

template<typename T>
gmacError_t SharedObject<T>::realloc(Mode &mode)
{
    void *device = NULL;
    // Allocate device and host memory
    gmacError_t ret = mode.malloc(&device, StateObject<T>::size_, paramPageSize);
    if(ret != gmacSuccess) {
        StateObject<T>::addr_ = NULL;
        return gmacErrorInsufficientDeviceMemory;
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
bool SharedObject<T>::isLocal() const
{
    return true;
}

template<typename T>
bool SharedObject<T>::isInAccelerator() const
{
    return true;
}

}}

#endif

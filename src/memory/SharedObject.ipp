#ifndef __MEMORY_SHAREDOBJECT_IPP
#define __MEMORY_SHAREDOBJECT_IPP

namespace gmac { namespace memory {

#ifdef USE_VM
#include "Bitmap.h"
#endif

template<typename T>
inline SharedObject<T>::SharedObject(size_t size, T init) :
    StateObject<T>(size),
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

#ifdef USE_MMAP
    StateObject<T>::addr_ = StateObject<T>::map(device, size);
#else
    StateObject<T>::addr_ = StateObject<T>::map(NULL, size);
#endif

    if(StateObject<T>::addr_ == NULL) {
        owner_->free(device);
        return;
    }

    trace("Creating Shared Object %p (%zd bytes)", StateObject<T>::addr_, StateObject<T>::size_);
    // Create memory blocks
    accBlock_ = new AcceleratorBlock(*owner_, device, StateObject<T>::size_);
    setupSystem(init);
}

template<typename T>
inline SharedObject<T>::~SharedObject()
{
    if(StateObject<T>::addr_ == NULL) { return; }
    void *devAddr = accBlock_->addr();
    delete accBlock_;
    StateObject<T>::unmap(StateObject<T>::addr_, StateObject<T>::size_);
    owner_->free(devAddr);
#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->dirtyBitmap();
    bitmap.removeRange(devAddr, StateObject<T>::size_);
#endif

    trace("Destroying Shared Object %p (%zd bytes)", StateObject<T>::addr_);
}

template<typename T>
inline void *SharedObject<T>::device(void *addr) const
{
    off_t offset = (unsigned long)addr - (unsigned long)StateObject<T>::addr_;
    void *ret = (uint8_t *)accBlock_->addr() + offset;
    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toHost(Block &block, void *hostAddr) const
{
    off_t off = (uint8_t *)block.addr() - (uint8_t *)StateObject<T>::addr_;
    gmacError_t ret;
    if (hostAddr == NULL) {
        ret = accBlock_->toHost(off, block);
    } else {
        ret = accBlock_->toHost(off, hostAddr, block.size());
    }

    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toDevice(Block &block) const
{
    off_t off = (uint8_t *)block.addr() - (uint8_t *)StateObject<T>::addr_;
    gmacError_t ret = accBlock_->toDevice(off, block);
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
bool SharedObject<T>::local() const
{
    return true;
}

}}

#endif

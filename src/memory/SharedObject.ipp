#ifndef __MEMORY_SHAREDOBJECT_IPP
#define __MEMORY_SHAREDOBJECT_IPP

namespace gmac { namespace memory {

#ifdef USE_VM
#include "Bitmap.h"
#endif

template<typename T>
inline SharedObject<T>::SharedObject(size_t size, T init) :
    StateObject<T>(size),
    _owner(&Mode::current()),
    _accBlock(NULL)
{
    gmacError_t ret = gmacSuccess;
    void *device = NULL;
    // Allocate device and host memory
    ret = _owner->malloc(&device, size, paramPageSize);
    if(ret != gmacSuccess) {
        StateObject<T>::addr_ = NULL;
        return;
    }
#ifdef USE_VM
    vm::Bitmap &bitmap = _owner->dirtyBitmap();
    bitmap.newRange(device, size);
#endif

#ifdef USE_MMAP
    StateObject<T>::addr_ = StateObject<T>::map(device, size);
#else
    StateObject<T>::addr_ = StateObject<T>::map(NULL, size);
#endif

    if(StateObject<T>::addr_ == NULL) {
        _owner->free(device);
        return;
    }

    trace("Creating Shared Object %p (%zd bytes)", StateObject<T>::addr_, StateObject<T>::size_);
    // Create memory blocks
    _accBlock = new AcceleratorBlock(*_owner, device, StateObject<T>::size_);
    setupSystem(init);
}

template<typename T>
inline SharedObject<T>::~SharedObject()
{
    if(StateObject<T>::addr_ == NULL) { return; }
    void *devAddr = _accBlock->addr();
    delete _accBlock;
    StateObject<T>::unmap(StateObject<T>::addr_, StateObject<T>::size_);
    _owner->free(devAddr);
#ifdef USE_VM
    vm::Bitmap &bitmap = _owner->dirtyBitmap();
    bitmap.removeRange(devAddr, StateObject<T>::size_);
#endif

    trace("Destroying Shared Object %p (%zd bytes)", StateObject<T>::addr_);
}

template<typename T>
inline void *SharedObject<T>::device(void *addr) const
{
    off_t offset = (unsigned long)addr - (unsigned long)StateObject<T>::addr_;
    void *ret = (uint8_t *)_accBlock->addr() + offset;
    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toHost(Block &block, void *hostAddr) const
{
    off_t off = (uint8_t *)block.addr() - (uint8_t *)StateObject<T>::addr_;
    gmacError_t ret;
    if (hostAddr == NULL) {
        ret = _accBlock->toHost(off, block);
    } else {
        ret = _accBlock->toHost(off, hostAddr, block.size());
    }

    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toDevice(Block &block) const
{
    off_t off = (uint8_t *)block.addr() - (uint8_t *)StateObject<T>::addr_;
    gmacError_t ret = _accBlock->toDevice(off, block);
    return ret;
}

template<typename T>
gmacError_t SharedObject<T>::free()
{
#ifdef USE_VM
    vm::Bitmap &bitmap = _owner->dirtyBitmap();
    bitmap.removeRange(_accBlock->addr(), StateObject<T>::size_);
#endif
    gmacError_t ret;
    ret = _owner->free(_accBlock->addr());
    delete _accBlock;
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

    trace("Reallocating object %p -> %p\n", _accBlock->addr(), device);

#ifdef USE_VM
    vm::Bitmap &bitmap = _owner->dirtyBitmap();
    bitmap.newRange(device, StateObject<T>::size_);
#endif
    _owner = &mode;
    _accBlock = new AcceleratorBlock(mode, device, StateObject<T>::size_);
    return gmacSuccess;
}

}}

#endif

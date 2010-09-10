#ifndef __MEMORY_SHAREDOBJECT_IPP
#define __MEMORY_SHAREDOBJECT_IPP

namespace gmac { namespace memory {

#ifdef USE_VM
#include "Bitmap.h"
#endif

template<typename T>
inline SharedObject<T>::SharedObject(size_t size, T init) :
    StateObject<T>(size),
    __owner(Mode::current()),
    accelerator(NULL)
{
    gmacError_t ret = gmacSuccess;
    void *device = NULL;
    // Allocate device and host memory
    ret = __owner->malloc(&device, size, paramPageSize);
    if(ret != gmacSuccess) {
        StateObject<T>::__addr = NULL;
        return;
    }
#ifdef USE_VM
    Mode *mode = Mode::current();
    vm::Bitmap &bitmap = mode->dirtyBitmap();
    bitmap.newRange(device, size);
#endif
    StateObject<T>::__addr = StateObject<T>::map(device, size);
    if(StateObject<T>::__addr == NULL) {
        __owner->free(device);
        return;
    }

    trace("Creating Shared Object %p (%zd bytes)", StateObject<T>::__addr, StateObject<T>::__size);
    // Create memory blocks
    accelerator = new AcceleratorBlock(__owner, device, StateObject<T>::__size);
    setupSystem(init);
}

template<typename T>
inline SharedObject<T>::~SharedObject()
{
    StateObject<T>::lockWrite();
    if(StateObject<T>::__addr == NULL) { StateObject<T>::unlock(); return; }
    void *devAddr = accelerator->addr();
    delete accelerator;
    StateObject<T>::unmap(StateObject<T>::__addr, StateObject<T>::__size);
    __owner->free(devAddr);
#ifdef USE_VM
    Mode *mode = Mode::current();
    vm::Bitmap &bitmap = mode->dirtyBitmap();
    bitmap.removeRange(devAddr, StateObject<T>::__size);
#endif

    trace("Destroying Shared Object %p (%zd bytes)", StateObject<T>::__addr);
    StateObject<T>::unlock();
}

template<typename T>
inline void *SharedObject<T>::device(void *addr)
{
    StateObject<T>::lockRead();
    off_t offset = (unsigned long)addr - (unsigned long)StateObject<T>::__addr;
    void *ret = (uint8_t *)accelerator->addr() + offset;
    StateObject<T>::unlock();
    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::acquire(Block *block)
{
    StateObject<T>::lockRead();
    off_t off = (uint8_t *)block->addr() - (uint8_t *)StateObject<T>::__addr;
    gmacError_t ret = accelerator->get(off, block);
    StateObject<T>::unlock();
    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::release(Block *block)
{
    StateObject<T>::lockRead();
    off_t off = (uint8_t *)block->addr() - (uint8_t *)StateObject<T>::__addr;
    gmacError_t ret = accelerator->put(off, block);
    StateObject<T>::unlock();
    return ret;
}

}}

#endif

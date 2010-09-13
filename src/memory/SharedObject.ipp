#ifndef __MEMORY_SHAREDOBJECT_IPP
#define __MEMORY_SHAREDOBJECT_IPP

namespace gmac { namespace memory {

#ifdef USE_VM
#include "Bitmap.h"
#endif

template<typename T>
inline SharedObject<T>::SharedObject(size_t size, T init) :
    StateObject<T>(size),
    _owner(Mode::current()),
    accelerator(NULL)
{
    gmacError_t ret = gmacSuccess;
    void *device = NULL;
    // Allocate device and host memory
    ret = _owner->malloc(&device, size, paramPageSize);
    if(ret != gmacSuccess) {
        StateObject<T>::_addr = NULL;
        return;
    }
#ifdef USE_VM
    vm::Bitmap &bitmap = _owner->dirtyBitmap();
    bitmap.newRange(device, size);
#endif
    StateObject<T>::_addr = StateObject<T>::map(device, size);
    if(StateObject<T>::_addr == NULL) {
        _owner->free(device);
        return;
    }

    trace("Creating Shared Object %p (%zd bytes)", StateObject<T>::_addr, StateObject<T>::_size);
    // Create memory blocks
    accelerator = new AcceleratorBlock(_owner, device, StateObject<T>::_size);
    setupSystem(init);
}

template<typename T>
inline SharedObject<T>::~SharedObject()
{
    if(StateObject<T>::_addr == NULL) { return; }
    void *devAddr = accelerator->addr();
    delete accelerator;
    StateObject<T>::unmap(StateObject<T>::_addr, StateObject<T>::_size);
    _owner->free(devAddr);
#ifdef USE_VM
    vm::Bitmap &bitmap = _owner->dirtyBitmap();
    bitmap.removeRange(devAddr, StateObject<T>::_size);
#endif

    trace("Destroying Shared Object %p (%zd bytes)", StateObject<T>::_addr);
}

template<typename T>
inline void *SharedObject<T>::device(void *addr) const
{
    off_t offset = (unsigned long)addr - (unsigned long)StateObject<T>::_addr;
    void *ret = (uint8_t *)accelerator->addr() + offset;
    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toHost(Block *block) const
{
    off_t off = (uint8_t *)block->addr() - (uint8_t *)StateObject<T>::_addr;
    gmacError_t ret = accelerator->get(off, block);
    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toDevice(Block *block) const
{
    off_t off = (uint8_t *)block->addr() - (uint8_t *)StateObject<T>::_addr;
    gmacError_t ret = accelerator->put(off, block);
    return ret;
}

template<typename T>
gmacError_t SharedObject<T>::move(Mode *mode)
{
    return gmacSuccess;
}

}}

#endif

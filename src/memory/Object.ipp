#ifndef __MEMORY_OBJECT_IPP
#define __MEMORY_OBJECT_IPP

#include <kernel/Mode.h>
#include <kernel/Context.h>

namespace gmac { namespace memory {


inline
Object::Object(void *__addr, size_t __size) :
    RWLock(paraver::LockObject),
    __addr(__addr),
    __size(__size),
    __owner(Mode::current())
{ }

template<typename T>
inline SharedObject<T>::SharedObject(size_t size) :
    Object(NULL, size),
    __accelerator(NULL)
{
    gmacError_t ret = gmacSuccess;
    void *device = NULL;
    // Allocate device and host memory
    ret = __owner->malloc(&device, size);
    if(ret != gmacSuccess) {
        __addr = NULL;
        return;
    }
    __addr = map(device, size);
    if(__addr == NULL) __owner->free(device);
    // Create memory blocks
    __accelerator = new AcceleratorBlock(__owner, device, __size);
    uint8_t *ptr = (uint8_t *)__addr;
    trace("Creating Shared Object %p (%zd bytes)", ptr, size);
    for(size_t i = 0; i < __size; i += paramPageSize, ptr += paramPageSize) {
        __system.insert(new SystemBlock<T>(ptr, paramPageSize));
    }
}

template<typename T>
inline SharedObject<T>::~SharedObject()
{
    if(__addr == NULL) return;
    // Clean all system blocks
    typename SystemSet::const_iterator i;
    for(i = __system.begin(); i != __system.end(); i++)
        delete (*i);
    __system.clear();
    void *device = __accelerator->addr();
    delete __accelerator;
    
    unmap(__addr, __size);
    __owner->free(device);
}

template<typename T>
inline void * SharedObject<T>::device() const 
{
    return __accelerator->addr();
}


}}

#endif

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
    Object(NULL, __size)
{
    gmacError_t ret = gmacSuccess;
    void *device = NULL;
    // Allocate device and host memory
    ret = __owner->context().malloc(&device, size);
    __addr = map(device, size);
    if(__addr == NULL) __owner->context().free(device);
    // Create memory blocks
    __accelerator = new AcceleratorBlock(__owner, device, __size);
    uint8_t *ptr = (uint8_t *)__addr;
    for(size_t i = 0; i < __size; i += paramPageSize, ptr += paramPageSize) {
        __system.insert(new SystemBlock<T>(ptr, paramPageSize));
    }
}

template<typename T>
inline SharedObject<T>::~SharedObject()
{
    // Clean all system blocks
    typename SystemSet::const_iterator i;
    for(i = __system.begin(); i != __system.end(); i++)
        delete (*i);
    __system.clear();
    void *device = __accelerator->addr();
    delete __accelerator;
    
    unmap(__addr, __size);
    __owner->context().free(device);
}

template<typename T>
inline void * SharedObject<T>::device() const 
{
    return __accelerator->addr();
}


}}

#endif

#ifndef __MEMORY_OBJECT_IPP
#define __MEMORY_OBJECT_IPP

#include <kernel/Mode.h>
#include <kernel/Context.h>

namespace gmac { namespace memory {


inline
Object::Object(void *__addr, size_t __size) :
    Lock(paraver::LockObject),
    __addr(__addr),
    __size(__size),
    __owner(Mode::current())
{ }

template<typename T>
inline SharedObject<T>::SharedObject(size_t size, T init) :
    Object(NULL, size),
    accelerator(NULL)
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
    accelerator = new AcceleratorBlock(__owner, device, __size);
    uint8_t *ptr = (uint8_t *)__addr;
    trace("Creating Shared Object %p (%zd bytes)", ptr, size);
    for(size_t i = 0; i < __size; i += paramPageSize, ptr += paramPageSize) {
        systemMap.insert(typename SystemMap::value_type(
            ptr + paramPageSize,
            new SystemBlock<T>(ptr, paramPageSize, init)));
    }
}

template<typename T>
inline SharedObject<T>::~SharedObject()
{
    if(__addr == NULL) return;
    // Clean all system blocks
    typename SystemMap::const_iterator i;
    for(i = systemMap.begin(); i != systemMap.end(); i++)
        delete i->second;
    systemMap.clear();
    void *device = accelerator->addr();
    delete accelerator;
    
    unmap(__addr, __size);
    __owner->free(device);
}

template<typename T>
inline void * SharedObject<T>::device() const 
{
    return accelerator->addr();
}

template<typename T>
inline void *SharedObject<T>::device(void *addr) const
{
    off_t offset = (unsigned long)addr - (unsigned long)__addr;
    return (uint8_t *)accelerator->addr() + offset;
}

template<typename T>
inline SystemBlock<T> *SharedObject<T>::findBlock(void *addr) 
{
    SystemBlock<T> *ret = NULL;
    typename SystemMap::const_iterator block = systemMap.upper_bound(addr);
    if(block != systemMap.end()) ret = block->second;
    return ret;
}

template<typename T>
inline void SharedObject<T>::state(T s)
{
    typename SystemMap::const_iterator i;
    for(i = systemMap.begin(); i != systemMap.end(); i++)
        i->second->state(s);
}

}}

#endif

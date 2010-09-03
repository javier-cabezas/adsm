#ifndef __MEMORY_OBJECT_IPP
#define __MEMORY_OBJECT_IPP

#include <kernel/Mode.h>
#include <kernel/Context.h>

namespace gmac { namespace memory {

inline Object::Object(void *__addr, size_t __size) :
    Lock(paraver::LockObject),
    __addr(__addr),
    __size(__size)
{ }

template<typename T>
inline StateObject<T>::StateObject(size_t size) :
    Object(NULL, size)
{ }

template<typename T>
inline StateObject<T>::~StateObject()
{
    // Clean all system blocks
    typename SystemMap::const_iterator i;
    for(i = systemMap.begin(); i != systemMap.end(); i++)
        delete i->second;
    systemMap.clear();
}

template<typename T>
inline void StateObject<T>::setupSystem(T init)
{
    uint8_t *ptr = (uint8_t *)__addr;
    for(size_t i = 0; i < __size; i += paramPageSize, ptr += paramPageSize) {
        systemMap.insert(typename SystemMap::value_type(
            ptr + paramPageSize,
            new SystemBlock<T>(ptr, paramPageSize, init)));
    }
}

template<typename T>
inline SystemBlock<T> *StateObject<T>::findBlock(void *addr) 
{
    SystemBlock<T> *ret = NULL;
    typename SystemMap::const_iterator block = systemMap.upper_bound(addr);
    if(block != systemMap.end()) ret = block->second;
    return ret;
}

template<typename T>
inline void StateObject<T>::state(T s)
{
    typename SystemMap::const_iterator i;
    for(i = systemMap.begin(); i != systemMap.end(); i++)
        i->second->state(s);
}




template<typename T>
inline SharedObject<T>::SharedObject(size_t size, T init) :
    StateObject<T>(size),
    __owner(Mode::current()),
    accelerator(NULL)
{
    gmacError_t ret = gmacSuccess;
    void *device = NULL;
    // Allocate device and host memory
    ret = __owner->malloc(&device, size);
    if(ret != gmacSuccess) {
        StateObject<T>::__addr = NULL;
        return;
    }
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
    if(StateObject<T>::__addr == NULL) return;
    delete accelerator;
    void *__device = accelerator->addr();
    StateObject<T>::unmap(StateObject<T>::__addr, StateObject<T>::__size);
    __owner->free(__device);
}

template<typename T>
inline void *SharedObject<T>::device(void *addr) const
{
    off_t offset = (unsigned long)addr - (unsigned long)StateObject<T>::__addr;
    return (uint8_t *)accelerator->addr() + offset;
}

template<typename T>
inline gmacError_t SharedObject<T>::acquire(Block *block)
{
    off_t off = (uint8_t *)block->addr() - (uint8_t *)StateObject<T>::__addr;
    return accelerator->get(off, block);
}

template<typename T>
inline gmacError_t SharedObject<T>::release(Block *block)
{
    off_t off = (uint8_t *)block->addr() - (uint8_t *)StateObject<T>::__addr;
    return accelerator->put(off, block);
}


#ifndef USE_MMAP
template<typename T>
inline ReplicatedObject<T>::ReplicatedObject(size_t size, T init) :
    StateObject<T>(size)
{
    // This line might seem useless, but we first need to make sure that
    // the curren thread has an execution mode attached
    Mode *mode = gmac::Mode::current(); 
    trace("Creating Replicated Object (%zd bytes)", StateObject<T>::__size);
    if(proc->globalMalloc(*this, size) != gmacSuccess) {
        Object::fatal("Unable to create replicated object");
        StateObject<T>::__addr = NULL;
        return;
    }

    StateObject<T>::__addr = StateObject<T>::map(NULL, size);
    if(StateObject<T>::__addr == NULL) {
        proc->globalFree(*this);
        return;
    }

    setupSystem(init);
    trace("Replicated object create @ %p", StateObject<T>::__addr);
}

template<typename T>
inline ReplicatedObject<T>::~ReplicatedObject()
{
    if(StateObject<T>::__addr == NULL) return;
    proc->globalFree(*this);
    accelerator.clear();
    
    StateObject<T>::unmap(StateObject<T>::__addr, StateObject<T>::__size);
}

template<typename T>
inline void *ReplicatedObject<T>::device(void *addr) const
{
    off_t offset = (unsigned long)addr - (unsigned long)StateObject<T>::__addr;
    AcceleratorMap::const_iterator i = accelerator.find(gmac::Mode::current());
    Object::assertion(i != accelerator.end());
    
    return (uint8_t *)i->second->addr() + offset;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::acquire(Block *block)
{
    Object::fatal("Reacquiring ownership of a replicated object is forbiden");
    return gmacErrorInvalidValue;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::release(Block *block)
{
    off_t off = (uint8_t *)block->addr() - (uint8_t *)StateObject<T>::__addr;
    AcceleratorMap::iterator i;
    for(i = accelerator.begin(); i != accelerator.end(); i++) {
        i->second->put(off, block);
    }
    return gmacSuccess;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::addOwner(Mode *mode)
{
    void *device = NULL;
    gmacError_t ret;
    ret = mode->malloc(&device, StateObject<T>::__size);
    Object::cfatal(ret == gmacSuccess, "Unable to replicate Object");
    accelerator.insert(AcceleratorMap::value_type(mode,
            new AcceleratorBlock(mode, device, StateObject<T>::__size)));
    trace("Adding replicated object @ %p to mode %p", device, mode);
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::removeOwner(Mode *mode)
{
    AcceleratorMap::iterator i = accelerator.find(mode);
    Object::assertion(i != accelerator.end());
    accelerator.erase(i);
    gmacError_t ret = mode->free(i->second->addr());
    delete i->second;
    return ret;
}

#endif

}}

#endif

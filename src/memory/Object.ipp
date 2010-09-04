#ifndef __MEMORY_OBJECT_IPP
#define __MEMORY_OBJECT_IPP

#include <kernel/Mode.h>
#include <kernel/Context.h>

namespace gmac { namespace memory {

inline Object::Object(void *__addr, size_t __size) :
    RWLock(paraver::LockObject),
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
    lockWrite();
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
        size_t blockSize = ((__size - i) > paramPageSize) ? paramPageSize : (__size - i);
        systemMap.insert(typename SystemMap::value_type(
            ptr + blockSize,
            new SystemBlock<T>(ptr, blockSize, init)));
    }
}

template<typename T>
inline SystemBlock<T> *StateObject<T>::findBlock(void *addr) 
{
    SystemBlock<T> *ret = NULL;
    lockRead();
    typename SystemMap::const_iterator block = systemMap.upper_bound(addr);
    if(block != systemMap.end()) ret = block->second;
    unlock();
    return ret;
}

template<typename T>
inline void StateObject<T>::state(T s)
{
    typename SystemMap::const_iterator i;
    lockWrite();
    for(i = systemMap.begin(); i != systemMap.end(); i++)
        i->second->state(s);
    unlock();
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
    StateObject<T>::lockWrite();
    if(StateObject<T>::__addr == NULL) { StateObject<T>::unlock(); return; }
    void *__device = accelerator->addr();
    delete accelerator;
    StateObject<T>::unmap(StateObject<T>::__addr, StateObject<T>::__size);
    __owner->free(__device);
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
    if(StateObject<T>::__addr == NULL) { return; }
    proc->globalFree(*this);
    StateObject<T>::lockWrite();
    accelerator.clear();
    StateObject<T>::unmap(StateObject<T>::__addr, StateObject<T>::__size);
    StateObject<T>::unlock();
}

template<typename T>
inline void *ReplicatedObject<T>::device(void *addr)
{
    StateObject<T>::lockRead();
    off_t offset = (unsigned long)addr - (unsigned long)StateObject<T>::__addr;
    typename AcceleratorMap::const_iterator i = accelerator.find(gmac::Mode::current());
    Object::assertion(i != accelerator.end());
    void *ret = (uint8_t *)i->second->addr() + offset;
    StateObject<T>::unlock();
    return ret;
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
    gmacError_t ret = gmacSuccess;
    StateObject<T>::lockRead();
    off_t off = (uint8_t *)block->addr() - (uint8_t *)StateObject<T>::__addr;
    typename AcceleratorMap::iterator i;
    for(i = accelerator.begin(); i != accelerator.end(); i++) {
        gmacError_t tmp = i->second->put(off, block);
        if(tmp != gmacSuccess) ret = tmp;
    }
    StateObject<T>::unlock();
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::addOwner(Mode *mode)
{
    void *device = NULL;
    gmacError_t ret;
    ret = mode->malloc(&device, StateObject<T>::__size);
    Object::cfatal(ret == gmacSuccess, "Unable to replicate Object");

    StateObject<T>::lockWrite();
    AcceleratorBlock *dev = new AcceleratorBlock(mode, device, StateObject<T>::__size);
    accelerator.insert(typename AcceleratorMap::value_type(mode, dev));
    typename StateObject<T>::SystemMap::iterator i;
    for(i = StateObject<T>::systemMap.begin(); i != StateObject<T>::systemMap.end(); i++) {
        if(mode->requireUpdate(i->second) == false) continue;
        off_t off = (uint8_t *)i->second->addr() - (uint8_t *)StateObject<T>::__addr;
        dev->put(off, i->second);
    }
    StateObject<T>::unlock();
    trace("Adding replicated object @ %p to mode %p", device, mode);
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::removeOwner(Mode *mode)
{
    StateObject<T>::lockWrite();
    typename AcceleratorMap::iterator i = accelerator.find(mode);
    Object::assertion(i != accelerator.end());
    AcceleratorBlock *acc = i->second;
    accelerator.erase(i);
    gmacError_t ret = mode->free(acc->addr());
    delete acc;
    StateObject<T>::unlock();
    return ret;
}

#endif

}}

#endif

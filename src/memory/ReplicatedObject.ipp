#ifndef __MEMORY_REPLICATEDOBJECT_IPP
#define __MEMORY_REPLICATEDOBJECT_IPP

namespace gmac { namespace memory {

#ifndef USE_MMAP
template<typename T>
inline ReplicatedObject<T>::ReplicatedObject(size_t size, T init) :
    StateObject<T>(size)
{
    // This line might seem useless, but we first need to make sure that
    // the curren thread has an execution mode attached
    Mode *mode = gmac::Mode::current(); 
    trace("Creating Replicated Object (%zd bytes)", StateObject<T>::_size);
    if(proc->globalMalloc(*this, size) != gmacSuccess) {
        Object::Fatal("Unable to create replicated object");
        StateObject<T>::_addr = NULL;
        return;
    }

    StateObject<T>::_addr = StateObject<T>::map(NULL, size);
    if(StateObject<T>::_addr == NULL) {
        proc->globalFree(*this);
        return;
    }

    setupSystem(init);
    trace("Replicated object create @ %p", StateObject<T>::_addr);
}

template<typename T>
inline ReplicatedObject<T>::~ReplicatedObject()
{
    if(StateObject<T>::_addr == NULL) { return; }
    proc->globalFree(*this);
    StateObject<T>::lockWrite();
    accelerator.clear();
    StateObject<T>::unmap(StateObject<T>::_addr, StateObject<T>::_size);
    StateObject<T>::unlock();
}

template<typename T>
inline void *ReplicatedObject<T>::device(void *addr)
{
    StateObject<T>::lockRead();
    off_t offset = (unsigned long)addr - (unsigned long)StateObject<T>::_addr;
    typename AcceleratorMap::const_iterator i = accelerator.find(gmac::Mode::current());
    Object::assertion(i != accelerator.end());
    void *ret = (uint8_t *)i->second->addr() + offset;
    StateObject<T>::unlock();
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::toHost(Block *block)
{
    Object::Fatal("Modifications to ReplicatedObjects in the device are forbidden");
    return gmacErrorInvalidValue;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::toDevice(Block *block)
{
    gmacError_t ret = gmacSuccess;
    StateObject<T>::lockRead();
    off_t off = (uint8_t *)block->addr() - (uint8_t *)StateObject<T>::_addr;
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
    void *devAddr = NULL;
    gmacError_t ret;
    ret = mode->malloc(&devAddr, StateObject<T>::_size, paramPageSize);
    Object::CFatal(ret == gmacSuccess, "Unable to replicate Object");

    StateObject<T>::lockWrite();
    AcceleratorBlock *dev = new AcceleratorBlock(mode, devAddr, StateObject<T>::_size);
    accelerator.insert(typename AcceleratorMap::value_type(mode, dev));
    typename StateObject<T>::SystemMap::iterator i;
    for(i = StateObject<T>::systemMap.begin(); i != StateObject<T>::systemMap.end(); i++) {
        if(mode->requireUpdate(i->second) == false) continue;
        off_t off = (uint8_t *)i->second->addr() - (uint8_t *)StateObject<T>::_addr;
        dev->put(off, i->second);
    }
#ifdef USE_VM
    vm::Bitmap & bitmap = mode->dirtyBitmap();
    bitmap.newRange(devAddr, StateObject<T>::_size);
#endif

    StateObject<T>::unlock();
    trace("Adding replicated object @ %p to mode %p", devAddr, mode);
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
#ifdef USE_VM
    vm::Bitmap & bitmap = mode->dirtyBitmap();
    bitmap.removeRange(acc->addr(), StateObject<T>::_size);
#endif
    delete acc;
    StateObject<T>::unlock();
    return ret;
}

#endif

}}

#endif

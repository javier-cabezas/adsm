#ifndef GMAC_MEMORY_REPLICATEDOBJECT_IPP
#define GMAC_MEMORY_REPLICATEDOBJECT_IPP

#include "core/Process.h"

namespace gmac { namespace memory {

#ifndef USE_MMAP
template<typename T>
inline ReplicatedObject<T>::ReplicatedObject(size_t size, T init) :
    StateObject<T>(size)
{
    // This line might seem useless, but we first need to make sure that
    // the current thread has an execution mode attached
    Process &proc = gmac::Process::getInstance();
    Mode &mode = gmac::Mode::current(); 
    trace("Creating Replicated Object (%zd bytes)", StateObject<T>::size_);
    if(proc.globalMalloc(*this, size) != gmacSuccess) {
        Object::Fatal("Unable to create replicated object");
        StateObject<T>::addr_ = NULL;
        return;
    }

    StateObject<T>::addr_ = StateObject<T>::map(NULL, size);
    if(StateObject<T>::addr_ == NULL) {
        proc.globalFree(*this);
        return;
    }

    setupSystem(init);
    trace("Replicated object create @ %p", StateObject<T>::addr_);
}

template<typename T>
inline ReplicatedObject<T>::~ReplicatedObject()
{
    if(StateObject<T>::addr_ == NULL) { return; }
    Process &proc = gmac::Process::getInstance();
    proc.globalFree(*this);
    StateObject<T>::lockWrite();
    accelerator.clear();
    StateObject<T>::unmap(StateObject<T>::addr_, StateObject<T>::size_);
    StateObject<T>::unlock();
}

template<typename T>
inline void *ReplicatedObject<T>::device(void *addr) const
{
    StateObject<T>::lockRead();
    off_t offset = (unsigned long)addr - (unsigned long)StateObject<T>::addr_;
    typename AcceleratorMap::const_iterator i = accelerator.find(&gmac::Mode::current());
    Object::assertion(i != accelerator.end());
    void *ret = (uint8_t *)i->second->addr() + offset;
    StateObject<T>::unlock();
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::toHost(Block &block, void * hostAddr) const
{
    Object::Fatal("Modifications to ReplicatedObjects in the device are forbidden");
    return gmacErrorInvalidValue;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::toDevice(Block &block) const
{
    gmacError_t ret = gmacSuccess;
    StateObject<T>::lockRead();
    off_t off = (uint8_t *)block.addr() - (uint8_t *)StateObject<T>::addr_;
    typename AcceleratorMap::const_iterator i;
    for(i = accelerator.begin(); i != accelerator.end(); i++) {
        gmacError_t tmp = i->second->toDevice(off, block);
        if(tmp != gmacSuccess) ret = tmp;
    }
    StateObject<T>::unlock();
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::addOwner(Mode &mode)
{
    void *devAddr = NULL;
    gmacError_t ret;
    ret = mode.malloc(&devAddr, StateObject<T>::size_, paramPageSize);
    Object::CFatal(ret == gmacSuccess, "Unable to replicate Object");

    StateObject<T>::lockWrite();
    AcceleratorBlock *dev = new AcceleratorBlock(mode, devAddr, StateObject<T>::size_);
    accelerator.insert(typename AcceleratorMap::value_type(&mode, dev));
    typename StateObject<T>::SystemMap::iterator i;
    for(i = StateObject<T>::systemMap.begin(); i != StateObject<T>::systemMap.end(); i++) {
        if(mode.requireUpdate(*i->second) == false) continue;
        off_t off = (uint8_t *)i->second->addr() - (uint8_t *)StateObject<T>::addr_;
        dev->toDevice(off, *i->second);
    }
#ifdef USE_VM
    vm::Bitmap & bitmap = mode.dirtyBitmap();
    bitmap.newRange(devAddr, StateObject<T>::size_);
#endif

    StateObject<T>::unlock();
    trace("Adding replicated object @ %p to mode %p", devAddr, &mode);
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::removeOwner(Mode &mode)
{
    StateObject<T>::lockWrite();
    typename AcceleratorMap::iterator i = accelerator.find(&mode);
    Object::assertion(i != accelerator.end());
    AcceleratorBlock *acc = i->second;
    accelerator.erase(i);
    gmacError_t ret = mode.free(acc->addr());
#ifdef USE_VM
    vm::Bitmap & bitmap = mode.dirtyBitmap();
    bitmap.removeRange(acc->addr(), StateObject<T>::size_);
#endif
    delete acc;
    StateObject<T>::unlock();
    return ret;
}

#endif

}}

#endif

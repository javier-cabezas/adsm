#ifndef GMAC_MEMORY_REPLICATEDOBJECT_IPP
#define GMAC_MEMORY_REPLICATEDOBJECT_IPP

#include "core/Process.h"

namespace gmac { namespace memory {

#ifndef USE_MMAP
template<typename T>
inline ReplicatedObject<T>::ReplicatedObject(size_t size, T init) :
    StateObject<T>(size, init)
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
    StateObject<T>::unlock();
}

template<typename T>
inline void ReplicatedObject<T>::init()
{
    StateObject<T>::addr_ = StateObject<T>::map(NULL, StateObject<T>::size_);
    if(StateObject<T>::addr_ == NULL) {
        return;
    }

    StateObject<T>::setupSystem();

}

template<typename T>
inline void ReplicatedObject<T>::fini()
{
    StateObject<T>::lockWrite();
    StateObject<T>::unmap(StateObject<T>::addr_, StateObject<T>::size_);
    StateObject<T>::unlock();
}


template<typename T>
inline void *ReplicatedObject<T>::getAcceleratorAddr(void *addr) const
{
    StateObject<T>::lockRead();
    off_t offset = (unsigned long)addr - (unsigned long)StateObject<T>::addr_;
    typename AcceleratorMap::const_iterator i = accelerator.find(&gmac::Mode::current());
    Object::assertion(i != accelerator.end());
    void *ret = i->second->addr() + offset;
    StateObject<T>::unlock();
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::toHost(Block &block) const
{
    Object::Fatal("Modifications to ReplicatedObjects in the accelerator are forbidden");
    return gmacErrorInvalidValue;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::toHost(Block &block, unsigned blockOff, size_t count) const
{
    Object::Fatal("Modifications to ReplicatedObjects in the accelerator are forbidden");
    return gmacErrorInvalidValue;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::toHostPointer(Block &block, unsigned blockOff, void *ptr, size_t count) const
{
    Object::Fatal("Modifications to ReplicatedObjects in the accelerator are forbidden");
    return gmacErrorInvalidValue;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::toHostBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    Object::Fatal("Modifications to ReplicatedObjects in the accelerator are forbidden");
    return gmacErrorInvalidValue;
}


template<typename T>
inline gmacError_t ReplicatedObject<T>::toAccelerator(Block &block) const
{
    gmacError_t ret = gmacSuccess;
    StateObject<T>::lockRead();
    off_t off = block.addr() - StateObject<T>::addr();
    typename AcceleratorMap::const_iterator i;
    for(i = accelerator.begin(); i != accelerator.end(); i++) {
        AcceleratorBlock &accBlock = *i->second;
        gmacError_t tmp = accBlock.owner().copyToAccelerator(accBlock.addr() + off, block.addr(), block.size());
        if(tmp != gmacSuccess) ret = tmp;
    }
    StateObject<T>::unlock();
    return ret;
}


template<typename T>
inline gmacError_t ReplicatedObject<T>::toAccelerator(Block &block, unsigned blockOff, size_t count) const
{
    assertion(block.addr() + blockOff + count <= block.end());
    gmacError_t ret = gmacSuccess;
    StateObject<T>::lockRead();
    off_t off = block.addr() + blockOff - StateObject<T>::addr();
    typename AcceleratorMap::const_iterator i;
    for(i = accelerator.begin(); i != accelerator.end(); i++) {
        AcceleratorBlock &accBlock = *i->second;
        gmacError_t tmp = accBlock.owner().copyToAccelerator(accBlock.addr() + off, block.addr() + blockOff, count);
        if(tmp != gmacSuccess) ret = tmp;
    }
    StateObject<T>::unlock();
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::toAcceleratorFromPointer(Block &block, unsigned blockOff, const void *ptr, size_t count) const
{
    assertion(block.addr() + blockOff + count <= block.end());
    gmacError_t ret = gmacSuccess;
    StateObject<T>::lockRead();
    off_t off = block.addr() + blockOff - StateObject<T>::addr();
    typename AcceleratorMap::const_iterator i;
    for(i = accelerator.begin(); i != accelerator.end(); i++) {
        AcceleratorBlock &accBlock = *i->second;
        gmacError_t tmp = accBlock.owner().copyToAccelerator(accBlock.addr() + off, ptr, count);
        if(tmp != gmacSuccess) ret = tmp;
    }
    StateObject<T>::unlock();
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::toAcceleratorFromBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    assertion(block.addr() + blockOff + count <= block.end());
    assertion(buffer.addr() + bufferOff + count <= buffer.end());
    gmacError_t ret = gmacSuccess;
    StateObject<T>::lockRead();
    off_t off = block.addr() + blockOff - StateObject<T>::addr();
    typename AcceleratorMap::const_iterator i;
    for(i = accelerator.begin(); i != accelerator.end(); i++) {
        AcceleratorBlock &accBlock = *i->second;
        gmacError_t tmp = accBlock.owner().bufferToAccelerator(accBlock.addr() + off, buffer, bufferOff, count);
        if(tmp != gmacSuccess) ret = tmp;
    }
    StateObject<T>::unlock();
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObject<T>::addOwner(Mode &mode)
{
    void *accAddr = NULL;
    gmacError_t ret;
    ret = mode.malloc(&accAddr, StateObject<T>::size_, paramPageSize);
    Object::CFatal(ret == gmacSuccess, "Unable to replicate Object");

    StateObject<T>::lockWrite();
    AcceleratorBlock *acc = new AcceleratorBlock(mode, accAddr, StateObject<T>::size_);
    accelerator.insert(typename AcceleratorMap::value_type(&mode, acc));
    typename StateObject<T>::SystemMap::iterator i;
    for(i = StateObject<T>::systemMap.begin(); i != StateObject<T>::systemMap.end(); i++) {
        SystemBlock<T> &block = *i->second;
        if(mode.requireUpdate(block) == false) continue;
        off_t off = block.addr() - StateObject<T>::addr();
        gmacError_t tmp = acc->owner().copyToAccelerator(acc->addr() + off, block.addr(), block.size());
    }
#ifdef USE_VM
    vm::Bitmap & bitmap = mode.dirtyBitmap();
    bitmap.newRange(accAddr, StateObject<T>::size_);
#endif

    StateObject<T>::unlock();
    trace("Adding replicated object @ %p to mode %p", accAddr, &mode);
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

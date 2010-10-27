#ifndef GMAC_MEMORY_REPLICATEDOBJECT_IPP
#define GMAC_MEMORY_REPLICATEDOBJECT_IPP

#include "core/Process.h"

namespace gmac { namespace memory {

#ifndef USE_MMAP
template<typename T>
inline ReplicatedObjectImpl<T>::ReplicatedObjectImpl(size_t size, T init) :
    StateObject<T>(size, init)
{
    // This line might seem useless, but we first need to make sure that
    // the current thread has an execution mode attached
    Process &proc = gmac::Process::getInstance();
    Mode &mode = gmac::Mode::current(); 
	UNREFERENCED_PARAMETER(mode);
    trace("Creating Replicated Object (%zd bytes)", StateObject<T>::size_);
    if(proc.globalMalloc(*this, size) != gmacSuccess) {
        Object::Fatal("Unable to create replicated object");
        StateObject<T>::addr_ = NULL;
        return;
    }

    trace("Replicated object create @ %p", StateObject<T>::addr_);
}



template<typename T>
inline ReplicatedObjectImpl<T>::~ReplicatedObjectImpl()
{
    if(StateObject<T>::addr_ == NULL) { return; }
    Process &proc = gmac::Process::getInstance();
    proc.globalFree(*this);
    accelerators_.clear();
}

template<typename T>
inline void ReplicatedObjectImpl<T>::init()
{
    StateObject<T>::addr_ = StateObject<T>::map(NULL, StateObject<T>::size_);
    if(StateObject<T>::addr_ == NULL) {
        return;
    }

    StateObject<T>::setupSystem();

}

template<typename T>
inline void ReplicatedObjectImpl<T>::fini()
{
    StateObject<T>::unmap(StateObject<T>::addr_, StateObject<T>::size_);
}


template<typename T>
inline void *ReplicatedObjectImpl<T>::getAcceleratorAddr(void *addr) const
{
    off_t offset = (unsigned long)addr - (unsigned long)StateObject<T>::addr_;
    typename AcceleratorMap::const_iterator i = accelerators_.find(&gmac::Mode::current());
    Object::assertion(i != accelerators_.end());
    void *ret = i->second->addr() + offset;
    return ret;
}

template<typename T>
inline Mode &ReplicatedObjectImpl<T>::owner() const
{
    return gmac::Mode::current();
}

template<typename T>
inline gmacError_t ReplicatedObjectImpl<T>::toHost(Block &) const
{
    Object::Fatal("Modifications to ReplicatedObjectImpls in the accelerator are forbidden");
    return gmacErrorInvalidValue;
}

template<typename T>
inline gmacError_t ReplicatedObjectImpl<T>::toHost(Block &, unsigned, size_t) const
{
    Object::Fatal("Modifications to ReplicatedObjectImpls in the accelerator are forbidden");
    return gmacErrorInvalidValue;
}

template<typename T>
inline gmacError_t ReplicatedObjectImpl<T>::toHostPointer(Block &, unsigned, void *, size_t) const
{
    Object::Fatal("Modifications to ReplicatedObjectImpls in the accelerator are forbidden");
    return gmacErrorInvalidValue;
}

template<typename T>
inline gmacError_t ReplicatedObjectImpl<T>::toHostBuffer(Block &, unsigned, IOBuffer &, unsigned, size_t) const
{
    Object::Fatal("Modifications to ReplicatedObjectImpls in the accelerator are forbidden");
    return gmacErrorInvalidValue;
}


template<typename T>
inline gmacError_t ReplicatedObjectImpl<T>::toAccelerator(Block &block) const
{
    gmacError_t ret = gmacSuccess;
    off_t off = (off_t)(block.addr() - StateObject<T>::addr());
    typename AcceleratorMap::const_iterator i;
    for(i = accelerators_.begin(); i != accelerators_.end(); i++) {
        AcceleratorBlock &accBlock = *i->second;
        gmacError_t tmp = accBlock.owner().copyToAccelerator(accBlock.addr() + off, block.addr(), block.size());
        if(tmp != gmacSuccess) ret = tmp;
    }
    return ret;
}


template<typename T>
inline gmacError_t ReplicatedObjectImpl<T>::toAccelerator(Block &block, unsigned blockOff, size_t count) const
{
    assertion(block.addr() + blockOff + count <= block.end());
    gmacError_t ret = gmacSuccess;
    off_t off = (off_t)(block.addr() + blockOff - StateObject<T>::addr());
    typename AcceleratorMap::const_iterator i;
    for(i = accelerators_.begin(); i != accelerators_.end(); i++) {
        AcceleratorBlock &accBlock = *i->second;
        gmacError_t tmp = accBlock.owner().copyToAccelerator(accBlock.addr() + off, block.addr() + blockOff, count);
        if(tmp != gmacSuccess) ret = tmp;
    }
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObjectImpl<T>::toAcceleratorFromPointer(Block &block, unsigned blockOff, const void *ptr, size_t count) const
{
    assertion(block.addr() + blockOff + count <= block.end());
    gmacError_t ret = gmacSuccess;
    off_t off = (off_t)(block.addr() + blockOff - StateObject<T>::addr());
    typename AcceleratorMap::const_iterator i;
    for(i = accelerators_.begin(); i != accelerators_.end(); i++) {
        AcceleratorBlock &accBlock = *i->second;
        gmacError_t tmp = accBlock.owner().copyToAccelerator(accBlock.addr() + off, ptr, count);
        if(tmp != gmacSuccess) ret = tmp;
    }
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObjectImpl<T>::toAcceleratorFromBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    assertion(block.addr() + blockOff + count <= block.end());
    assertion(buffer.addr() + bufferOff + count <= buffer.end());
    gmacError_t ret = gmacSuccess;
    off_t off = (off_t)(block.addr() + blockOff - StateObject<T>::addr());
    typename AcceleratorMap::const_iterator i;
    for(i = accelerators_.begin(); i != accelerators_.end(); i++) {
        AcceleratorBlock &accBlock = *i->second;
        gmacError_t tmp = accBlock.owner().bufferToAccelerator(accBlock.addr() + off, buffer, count, bufferOff);
        if(tmp != gmacSuccess) ret = tmp;
    }
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObjectImpl<T>::addOwner(Mode &mode)
{
    util::RWLock::lockWrite();
    void *accAddr = NULL;
    gmacError_t ret;
    ret = mode.malloc(&accAddr, StateObject<T>::size_, (unsigned)paramPageSize);
    Object::CFatal(ret == gmacSuccess, "Unable to replicate Object");

    AcceleratorBlock *acc = new AcceleratorBlock(mode, accAddr, StateObject<T>::size_);
    accelerators_.insert(typename AcceleratorMap::value_type(&mode, acc));
    typename StateObject<T>::SystemMap::iterator i;
    for(i = StateObject<T>::systemMap.begin(); i != StateObject<T>::systemMap.end(); i++) {
        SystemBlock<T> &block = *i->second;
        if(mode.requireUpdate(block) == false) continue;
        off_t off = (off_t)(block.addr() - StateObject<T>::addr());
        gmacError_t tmp = acc->owner().copyToAccelerator(acc->addr() + off, block.addr(), block.size());
		if(tmp != gmacSuccess) {
            util::RWLock::unlock();
            return tmp;
        }
    }
#ifdef USE_VM
    vm::Bitmap & bitmap = mode.dirtyBitmap();
    bitmap.newRange(accAddr, StateObject<T>::size_);
#endif

    trace("Adding replicated object @ %p to mode %p", accAddr, &mode);
    util::RWLock::unlock();
    return ret;
}

template<typename T>
inline gmacError_t ReplicatedObjectImpl<T>::removeOwner(Mode &mode)
{
    util::RWLock::lockWrite();
    typename AcceleratorMap::iterator i = accelerators_.find(&mode);
    Object::assertion(i != accelerators_.end());
    AcceleratorBlock *acc = i->second;
    accelerators_.erase(i);
    gmacError_t ret = mode.free(acc->addr());
#ifdef USE_VM
    vm::Bitmap & bitmap = mode.dirtyBitmap();
    bitmap.removeRange(acc->addr(), StateObject<T>::size_);
#endif
    delete acc;
    util::RWLock::unlock();
    return ret;
}

#endif

template <typename T>
inline bool
ReplicatedObjectImpl<T>::isLocal() const
{
    return false;
}

template <typename T>
inline bool
ReplicatedObjectImpl<T>::isInAccelerator() const
{
    return true;
}

}}

#endif

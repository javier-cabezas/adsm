#ifndef GMAC_MEMORY_DISTRIBUTEDOBJECT_IMPL_H_
#define GMAC_MEMORY_DISTRIBUTEDOBJECT_IMPL_H_

#include "core/Mode.h"
#include "memory/DistributedBlock.h"

namespace __impl { namespace memory {

template<typename T>
inline DistributedObject<T>::DistributedObject(Protocol &protocol, core::Mode &owner,
											   hostptr_t cpuAddr, size_t size, T init) :
    Object(cpuAddr, size)
{
    // Allocate memory (if necessary)
    if(addr_ == NULL)
        addr_ = Memory::map(NULL, size, GMAC_PROT_READWRITE);
    if(addr_ == NULL) return;

    // Create a shadow mapping for the host memory
    shadow_ = Memory::shadow(addr_, size_);

    accptr_t acceleratorAddr = NULL;
    // Allocate accelerator memory
    gmacError_t ret = 
        owner.malloc(acceleratorAddr, size, unsigned(paramPageSize));
    if(ret == gmacSuccess) valid_ = true;

    // Populate the block-set
    acceleratorAddr_.insert(AcceleratorMap::value_type(&owner, acceleratorAddr));
    hostptr_t mark = addr_;
    int offset = 0;
    while(size > 0) {
        size_t blockSize = (size > paramPageSize) ? paramPageSize : size;
        mark += blockSize;
        blocks_.insert(BlockMap::value_type(mark,
			new DistributedBlock<T>(protocol, owner, addr_ + offset, shadow_ + offset,
			acceleratorAddr + offset, blockSize, init)));
        size -= blockSize;
        offset += int(blockSize);
    }
    TRACE(GLOBAL, "Creating Distributed Object @ %p : shadow @ %p : accelerator @ %p) ", 
        addr_, shadow_, (void *) acceleratorAddr);
}


template<typename T>
inline DistributedObject<T>::~DistributedObject()
{
    AcceleratorMap::iterator i;
    for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++)
        i->first->free(i->second);
    Memory::unshadow(shadow_, size_);
    TRACE(GLOBAL, "Destroying Distributed Object @ %p", addr_);
}

template<typename T>
inline accptr_t DistributedObject<T>::acceleratorAddr(const hostptr_t addr) const
{
	accptr_t ret = NULL;
	lockRead();
	BlockMap::const_iterator i = blocks_.upper_bound(addr);
	if(i != blocks_.end()) {
		ret = i->second->acceleratorAddr(addr);
	}
	unlock();
	return ret;
}

template<typename T>
inline core::Mode &DistributedObject<T>::owner(const hostptr_t addr) const
{
	lockRead();
	BlockMap::const_iterator i = blocks_.upper_bound(addr);
	ASSERTION(i != blocks_.end());
	core::Mode &ret = i->second->owner();
	unlock();
	return ret;
}


template<typename T>
inline gmacError_t DistributedObject<T>::addOwner(core::Mode &mode)
{
    // Make sure that we do not add the same owner twice
    lockRead();
    bool alreadyOwned = (acceleratorAddr_.find(&mode) != acceleratorAddr_.end());
    unlock();
    if(alreadyOwned) return gmacSuccess;

    accptr_t acceleratorAddr = NULL;
    gmacError_t ret = 
		mode.malloc(acceleratorAddr, size_, unsigned(paramPageSize));
    if(ret != gmacSuccess) return ret;

    lockWrite();
    acceleratorAddr_.insert(AcceleratorMap::value_type(&mode, acceleratorAddr));
    BlockMap::iterator i;
    for(i = blocks_.begin(); i != blocks_.end(); i++) {
        ptroff_t offset = ptroff_t(i->second->addr() - addr_);
        DistributedBlock<T> &block = dynamic_cast<DistributedBlock<T> &>(*i->second);
        block.addOwner(mode, acceleratorAddr + offset);
        
    }
	unlock();
	return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedObject<T>::removeOwner(const core::Mode &mode)
{
	lockWrite();
    AcceleratorMap::iterator i = acceleratorAddr_.find((core::Mode *)&mode);
    if(i != acceleratorAddr_.end()) {
        BlockMap::iterator j;
        for(j = blocks_.begin(); j != blocks_.end(); j++) {
            DistributedBlock<T> &block = dynamic_cast<DistributedBlock<T> &>(*j->second);
            block.removeOwner(*i->first);
        }
        i->first->free(i->second);
        acceleratorAddr_.erase(i);
        if(acceleratorAddr_.empty()) Map::insertOrphan(*this);
    }
    unlock();
	return gmacSuccess;
}

template<typename T>
inline
gmacError_t DistributedObject<T>::mapToAccelerator()
{
    // TODO Fail
	return gmacSuccess;
}

template<typename T>
inline
gmacError_t DistributedObject<T>::unmapFromAccelerator()
{
    // TODO Fail
	return gmacSuccess;
}

}}

#endif

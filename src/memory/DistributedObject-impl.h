#ifndef GMAC_MEMORY_DISTRIBUTEDOBJECT_IMPL_H_
#define GMAC_MEMORY_DISTRIBUTEDOBJECT_IMPL_H_

#include "core/Mode.h"
#include "memory/DistributedBlock.h"

namespace __impl { namespace memory {

template<typename T>
inline DistributedObject<T>::DistributedObject(Protocol &protocol, core::Mode &owner,
											   void *cpuAddr, size_t size, T init) :
    Object(cpuAddr, size)
{
    // Allocate memory (if necessary)
	if(addr_ == NULL)
		addr_ = (uint8_t *)Memory::map(NULL, size, GMAC_PROT_READWRITE);
    if(addr_ == NULL) return;

    // Create a shadow mapping for the host memory
    shadow_ = (uint8_t *)Memory::shadow(addr_, size_);

    uint8_t *deviceAddr = NULL;
    // Allocate accelerator memory
    gmacError_t ret = 
		owner.malloc((void **)&deviceAddr, size, (unsigned)paramPageSize);
	if(ret == gmacSuccess) valid_ = true;

	// Populate the block-set
    deviceAddr_.insert(DeviceMap::value_type(&owner, deviceAddr));
	uint8_t *mark = addr_;
	unsigned offset = 0;
	while(size > 0) {
		size_t blockSize = (size > paramPageSize) ? paramPageSize : size;
		mark += blockSize;
		blocks_.insert(BlockMap::value_type(mark, 
			new DistributedBlock<T>(protocol, owner, addr_ + offset, shadow_ + offset,
			deviceAddr + offset, blockSize, init)));
		size -= blockSize;
		offset += unsigned(blockSize);
	}
    TRACE(GLOBAL, "Creating Distributed Object @ %p : shadow @ %p : device @ %p) ", 
        addr_, shadow_, deviceAddr);
}


template<typename T>
inline DistributedObject<T>::~DistributedObject()
{
    DeviceMap::iterator i;
    for(i = deviceAddr_.begin(); i != deviceAddr_.end(); i++)
        i->first->free(i->second);
    Memory::unshadow(shadow_, size_);
    TRACE(GLOBAL, "Destroying Distributed Object @ %p", addr_);
}

template<typename T>
inline void *DistributedObject<T>::deviceAddr(const void *addr) const
{
	void *ret = NULL;
	lockRead();
	BlockMap::const_iterator i = blocks_.upper_bound((uint8_t *)addr);
	if(i != blocks_.end()) {
		ret = i->second->deviceAddr(addr);
	}
	unlock();
	return ret;
}

template<typename T>
inline core::Mode &DistributedObject<T>::owner(const void *addr) const
{
	lockRead();
	BlockMap::const_iterator i = blocks_.upper_bound((uint8_t *)addr);
	ASSERTION(i != blocks_.end());
	core::Mode &ret = i->second->owner();
	unlock();
	return ret;
}


template<typename T>
inline bool DistributedObject<T>::addOwner(core::Mode &mode)
{
    // Make sure that we do not add the same owner twice
    lockRead();
    bool alreadyOwned = (deviceAddr_.find(&mode) != deviceAddr_.end());
    unlock();
    if(alreadyOwned) return true;

    uint8_t *deviceAddr = NULL;
    gmacError_t ret = 
		mode.malloc((void **)&deviceAddr, size_, (unsigned)paramPageSize);
    if(ret != gmacSuccess) return false;
    lockRead();
    BlockMap::iterator i;
    for(i = blocks_.begin(); i != blocks_.end(); i++) {
        unsigned offset = unsigned(i->second->addr() - addr_);
        DistributedBlock<T> &block = dynamic_cast<DistributedBlock<T> &>(*i->second);
        block.addOwner(mode, deviceAddr + offset);
        
    }
	unlock();
	return true;
}

template<typename T>
inline void DistributedObject<T>::removeOwner(const core::Mode &mode)
{
	lockWrite();
    DeviceMap::iterator i = deviceAddr_.find((core::Mode *)&mode);
    if(i != deviceAddr_.end()) {
        BlockMap::iterator j;
        for(j = blocks_.begin(); j != blocks_.end(); j++) {
            DistributedBlock<T> &block = dynamic_cast<DistributedBlock<T> &>(*j->second);
            block.removeOwner(*i->first);
        }
        i->first->free(i->second);
        deviceAddr_.erase(i);
        if(deviceAddr_.empty()) Map::insertOrphan(*this);
    }
    unlock();
	return;
}

}}

#endif

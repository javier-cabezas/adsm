#ifndef GMAC_MEMORY_DISTRIBUTEDOBJECT_IMPL_H_
#define GMAC_MEMORY_DISTRIBUTEDOBJECT_IMPL_H_

#include "core/Mode.h"
#include "memory/DistributedBlock.h"

namespace __impl { namespace memory {

template<typename T>
inline DistributedObject<T>::DistributedObject(Protocol &protocol, core::Mode &owner,
											   void *cpuAddr, size_t size, T init) :
    Object(cpuAddr, size),
	owner_(owner)
{
	if(addr_ == NULL) return;
    // Allocate accelerator memory
    gmacError_t ret = 
		owner.malloc((void **)&deviceAddr_, size, (unsigned)paramPageSize);
	if(ret == gmacSuccess) valid_ = true;

	// Populate the block-set
	uint8_t *mark = addr_;
	unsigned offset = 0;
	while(size > 0) {
		size_t blockSize = (size > paramPageSize) ? paramPageSize : size;
		mark += blockSize;
		blocks_.insert(BlockMap::value_type(mark, 
			new DistributedBlock<T>(protocol, owner_, addr_ + offset, shadow_ + offset,
			deviceAddr_ + offset, blockSize, init)));
		size -= blockSize;
		offset += unsigned(blockSize);
	}
}


template<typename T>
inline DistributedObject<T>::~DistributedObject()
{
	// If the object creation failed, this address will be NULL
    if(deviceAddr_ != NULL) owner_.free(deviceAddr_);
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
inline void DistributedObject<T>::addOwner(core::Mode &mode)
{
	// TODO: fill the logic
	return;
}

template<typename T>
inline void DistributedObject<T>::removeOwner(const core::Mode &mode)
{
	// TODO: fill the logic
	return;
}

}}

#endif
